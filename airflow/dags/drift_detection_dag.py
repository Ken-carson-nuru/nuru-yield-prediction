from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
import os
import socket
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
from loguru import logger

sys.path.insert(0, '/opt/airflow/dags/repo')

import mlflow

from src.storage import DataStorage
from config.settings import get_settings

settings = get_settings()
BUCKET = settings.S3_BUCKET_NAME

default_args = {
    "owner": "nuru",
    "start_date": days_ago(1),
    "retries": 2,
}

def _get_ds() -> str:
    d = os.environ.get("AIRFLOW_CTX_EXECUTION_DATE")
    if d:
        try:
            return pd.to_datetime(d).strftime("%Y-%m-%d")
        except Exception:
            return d.split("T")[0]
    return datetime.utcnow().strftime("%Y-%m-%d")


def _get_tracking_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        try:
            from urllib.parse import urlparse

            p = urlparse(uri)
            host = p.hostname
            if p.scheme in ("http", "https") and host:
                socket.gethostbyname(host)
                return uri
        except Exception:
            logger.warning(f"Invalid or unreachable MLFLOW_TRACKING_URI '{uri}', falling back to local path")
    return "/opt/airflow/dags/repo/mlruns"

def _read_parquet(storage: DataStorage, bucket: str, key: str) -> pd.DataFrame:
    r = storage.s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(BytesIO(r["Body"].read()))

def _list_daily_feature_keys(storage: DataStorage, bucket: str, base_prefix: str) -> list:
    pref = f"{base_prefix}/features/"
    try:
        resp = storage.s3_client.list_objects_v2(Bucket=bucket, Prefix=pref)
        contents = resp.get("Contents", [])
        return [o["Key"] for o in contents if o["Key"].endswith("/daily_features.parquet")]
    except Exception as e:
        logger.warning(f"List daily features failed: {e}")
        return []

def _extract_date_from_key(key: str) -> datetime:
    try:
        parts = key.split("/")
        ds = parts[-2]
        return pd.to_datetime(ds)
    except Exception:
        return None

def _psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    r = ref.dropna().astype(float)
    c = cur.dropna().astype(float)
    if len(r) == 0 or len(c) == 0:
        return 0.0
    qs = np.quantile(r, np.linspace(0, 1, bins + 1))
    qs[0] = -np.inf
    qs[-1] = np.inf
    r_counts = np.histogram(r, bins=qs)[0]
    c_counts = np.histogram(c, bins=qs)[0]
    r_prop = np.where(r_counts == 0, 1e-6, r_counts / max(r.size, 1))
    c_prop = np.where(c_counts == 0, 1e-6, c_counts / max(c.size, 1))
    return float(np.sum((c_prop - r_prop) * np.log(c_prop / r_prop)))

def _psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    r = ref.dropna().astype(str)
    c = cur.dropna().astype(str)
    if len(r) == 0 or len(c) == 0:
        return 0.0
    cats = list(set(r.unique().tolist() + c.unique().tolist()))
    r_counts = r.value_counts()
    c_counts = c.value_counts()
    r_prop = np.array([r_counts.get(k, 0) for k in cats], dtype=float)
    c_prop = np.array([c_counts.get(k, 0) for k in cats], dtype=float)
    r_prop = np.where(r_prop == 0, 1e-6, r_prop / max(r.size, 1))
    c_prop = np.where(c_prop == 0, 1e-6, c_prop / max(c.size, 1))
    return float(np.sum((c_prop - r_prop) * np.log(c_prop / r_prop)))

def _select_feature_cols(df: pd.DataFrame) -> list:
    ignore = set(["plot_id", "date", "planting_date", "latitude", "longitude"])
    return [c for c in df.columns if c not in ignore]

with DAG(
    dag_id="drift_detection_dag",
    default_args=default_args,
    schedule_interval="0 4 * * *",
    catchup=False,
    tags=["monitoring", "drift"],
):

    @task(multiple_outputs=True)
    def detect_drift() -> dict:
        storage = DataStorage()
        ds = _get_ds()
        base_prefix = settings.S3_BASE_PREFIX
        cur_key = f"{base_prefix}/features/{ds}/daily_features.parquet"
        try:
            cur_df = _read_parquet(storage, BUCKET, cur_key)
            logger.info(f"Loaded current daily features: s3://{BUCKET}/{cur_key}")
        except Exception as e:
            keys = _list_daily_feature_keys(storage, BUCKET, base_prefix)
            dated = [(k, _extract_date_from_key(k)) for k in keys]
            dated = [(k, d) for k, d in dated if d is not None]
            if not dated:
                logger.warning("No daily features available")
                return {"status": "no_data"}
            dated.sort(key=lambda x: x[1], reverse=True)
            cur_key = dated[0][0]
            cur_df = _read_parquet(storage, BUCKET, cur_key)
            ds = _extract_date_from_key(cur_key).strftime("%Y-%m-%d")

        keys = _list_daily_feature_keys(storage, BUCKET, base_prefix)
        cutoff = pd.to_datetime(ds) - timedelta(days=1)
        hist = [(k, _extract_date_from_key(k)) for k in keys]
        hist = [(k, d) for k, d in hist if d is not None and d <= cutoff]
        hist.sort(key=lambda x: x[1], reverse=True)
        ref_keys = [k for k, d in hist[:30]]
        if not ref_keys:
            logger.warning("No reference days available")
            return {"status": "no_reference"}

        dfs = []
        for k in ref_keys:
            try:
                dfs.append(_read_parquet(storage, BUCKET, k))
            except Exception:
                continue
        if not dfs:
            return {"status": "no_reference"}
        ref_df = pd.concat(dfs, ignore_index=True)

        cols = _select_feature_cols(cur_df)
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(cur_df[c])]
        cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(cur_df[c])]

        rows = []
        for c in numeric_cols:
            psi = _psi_numeric(ref_df[c], cur_df[c])
            miss_ref = float(ref_df[c].isna().mean())
            miss_cur = float(cur_df[c].isna().mean())
            rows.append({"feature": c, "type": "numeric", "psi": psi, "missing_ref": miss_ref, "missing_cur": miss_cur})
        for c in cat_cols:
            psi = _psi_categorical(ref_df[c], cur_df[c])
            miss_ref = float(ref_df[c].isna().mean())
            miss_cur = float(cur_df[c].isna().mean())
            rows.append({"feature": c, "type": "categorical", "psi": psi, "missing_ref": miss_ref, "missing_cur": miss_cur})

        report_df = pd.DataFrame(rows)
        def level(x):
            if x >= 0.25:
                return "high"
            if x >= 0.1:
                return "medium"
            return "low"
        report_df["level"] = report_df["psi"].apply(level)

        summary = {
            "ds": ds,
            "n_features": int(len(report_df)),
            "n_high": int((report_df["level"] == "high").sum()),
            "n_medium": int((report_df["level"] == "medium").sum()),
            "status": "ok",
        }

        # MLflow logging (ensure we always log somewhere, default local path if env not set)
        tracking_uri = _get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)

        # Ensure S3/MinIO artifact settings are present (mirrors training DAG defaults)
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ.get(
            "MLFLOW_S3_ENDPOINT_URL", os.environ.get("S3_ENDPOINT_URL", "http://minio:9000")
        )
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get(
            "S3_ACCESS_KEY", os.environ.get("MINIO_ROOT_USER", "minioadmin") or "minioadmin"
        )
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get(
            "S3_SECRET_KEY", os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin") or "minioadmin"
        )

        mlflow.set_experiment("DriftMonitoring")
        with mlflow.start_run(run_name=f"drift_{ds}"):
            mlflow.log_params({"ds": ds, "reference_days": len(ref_keys), "n_features": int(len(report_df))})
            mlflow.log_metrics({"n_high": summary["n_high"], "n_medium": summary["n_medium"]})
            top = report_df.sort_values("psi", ascending=False).head(20)
            tmp_csv = f"/tmp/drift_report_{ds}.csv"
            tmp_json = f"/tmp/drift_summary_{ds}.json"
            top.to_csv(tmp_csv, index=False)
            import json
            with open(tmp_json, "w") as f:
                json.dump(summary, f)
            mlflow.log_artifact(tmp_csv, artifact_path="drift")
            mlflow.log_artifact(tmp_json, artifact_path="drift")

        date_prefix = ds
        base = f"{settings.S3_BASE_PREFIX}/monitoring/drift/{date_prefix}"
        try:
            buf = BytesIO()
            report_df.to_parquet(buf, index=False)
            buf.seek(0)
            storage.s3_client.put_object(Bucket=BUCKET, Key=f"{base}/report.parquet", Body=buf.getvalue())
        except Exception:
            pass

        if summary["n_high"] > 0:
            summary["alert"] = "high_drift"
        else:
            summary["alert"] = "none"

        return {"summary": summary, "s3_prefix": f"s3://{BUCKET}/{base}"}

    @task(multiple_outputs=False)
    def finalize(res: dict) -> bool:
        logger.info(str(res))
        return True

    r = detect_drift()
    done = finalize(r)

