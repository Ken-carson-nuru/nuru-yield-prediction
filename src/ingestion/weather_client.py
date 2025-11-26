# src/weather_client.py

import requests
import time
from typing import List, Optional
from datetime import datetime
import pandas as pd
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from config.settings import get_settings
from config.schemas import PlotInput, WeatherDataPoint


class WeatherAPIClient:
    """
    Visual Crossing Weather API client with:
    - Retry w/ exponential backoff
    - Rate limiting handling
    - Validated response (Pydantic)
    - Thread-safe request session
    - Structured logging
    """

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.VISUAL_CROSSING_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NuruWeatherIngestion/2.0"})

    # ----------------------------------------------------
    # 1. Season → Date Range Mapping
    # ----------------------------------------------------
    def _get_season_date_range(self, season: str, year: int) -> tuple[str, str]:
        """
        Short Rains:  Sep 1 – Dec 31
        Long Rains:   Feb 1 – May 31
        """
        season = season.lower()

        if season == "short rains":
            start = f"{year}-09-01"
            end = f"{year}-12-31"

        elif season == "long rains":
            start = f"{year}-02-01"
            end = f"{year}-05-31"

        else:
            raise ValueError(f"Invalid season '{season}'. Must be 'Short Rains' or 'Long Rains'.")

        logger.info(f"Season '{season}' → date window: {start} → {end}")
        return start, end

    # ----------------------------------------------------
    # 2. API Request with Retry
    # ----------------------------------------------------
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(
            (requests.exceptions.RequestException, requests.exceptions.Timeout)
        ),
        before_sleep=lambda state: logger.warning(
            f"Retry {state.attempt_number}/5 after failure: {state.outcome.exception()}"
        )
    )
    def _make_api_request(self, lat: float, lon: float, start_date: str, end_date: str) -> dict:
        url = f"{self.base_url}/{lat},{lon}/{start_date}/{end_date}"
        params = {
            "unitGroup": "metric",
            "include": "days",
            "key": self.settings.VISUAL_CROSSING_API_KEY,
            "contentType": "json",
        }

        logger.debug(f"GET {url}  params={params}")

        response = self.session.get(
            url,
            params=params,
            timeout=self.settings.REQUEST_TIMEOUT,
        )

        # Handle rate limiting explicitly
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 60))
            logger.warning(f"Rate Limited (429). Sleeping {retry_after}s")
            time.sleep(retry_after)
            raise requests.exceptions.RequestException("Rate limit exceeded")

        response.raise_for_status()
        return response.json()

    # ----------------------------------------------------
    # 3. Single Plot Weather Fetch
    # ----------------------------------------------------
    def fetch_weather_for_plot(self, plot: PlotInput) -> List[WeatherDataPoint]:
        """Fetch weather data for a single plot."""

        start, end = self._get_season_date_range(plot.season, plot.year)
        logger.info(
            f"Fetching: plot_id={plot.plot_id}, lat={plot.latitude}, "
            f"lon={plot.longitude}, window={start}→{end}"
        )

        try:
            raw = self._make_api_request(plot.latitude, plot.longitude, start, end)
        except Exception as e:
            logger.error(f"API failure for plot {plot.plot_id}: {e}")
            raise

        if "days" not in raw or not raw["days"]:
            logger.error(f"No weather days returned for plot {plot.plot_id}")
            raise ValueError(f"No weather data found for plot_id {plot.plot_id}")

        validated = []
        for day in raw["days"]:
            try:
                item = WeatherDataPoint(
                    plot_id=plot.plot_id,
                    latitude=plot.latitude,
                    longitude=plot.longitude,
                    date=datetime.fromisoformat(day["datetime"]),
                    **day
                )
                validated.append(item)

            except Exception as e:
                logger.warning(
                    f"Failed day parsing for {plot.plot_id} @ {day.get('datetime')}: {e}"
                )
                continue

        logger.success(f"Plot {plot.plot_id} → {len(validated)} valid weather records")
        return validated

    # ----------------------------------------------------
    # 4. Batch Processing With ThreadPool
    # ----------------------------------------------------
    def fetch_weather_batch(
        self,
        plots: List[PlotInput],
        max_workers: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch weather for many plots concurrently.
        Returns a *validated*, flattened DataFrame.
        """

        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers = max_workers or self.settings.MAX_WORKERS
        logger.info(f"Batch weather fetch for {len(plots)} plots using {max_workers} workers")

        all_records = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(self.fetch_weather_for_plot, p): p for p in plots}

            for future in as_completed(future_map):
                plot = future_map[future]

                try:
                    records = future.result()
                    all_records.extend([r.model_dump() for r in records])
                except Exception as e:
                    logger.error(f"Plot {plot.plot_id} failed: {e}")
                    continue

        if not all_records:
            raise ValueError("No weather data collected for any plot.")

        df = pd.DataFrame(all_records)

        logger.success(
            f"Batch completed: {len(df)} total records, {df['plot_id'].nunique()} plots"
        )

        return df
