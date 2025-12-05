# src/ingestion/vegetation_indices.py
import ee
from loguru import logger


class VegetationIndicesCalculator:
    """Calculate vegetation indices from Sentinel-2 imagery"""

    BAND_MAPPING = {
        "BLUE": "B2",
        "GREEN": "B3",
        "RED": "B4",
        "RED_EDGE_1": "B5",
        "NIR": "B8",
        "SWIR_1": "B11"
    }

    def _select_band(self, image: ee.Image, band_key: str):
        """Prevent crashes if a band is missing"""
        band_name = self.BAND_MAPPING[band_key]
        try:
            if image.bandNames().contains(band_name).getInfo():
                return image.select(band_name)
            else:
                logger.warning(f"Band missing: {band_name}, using zero fallback")
                return ee.Image.constant(0).rename(band_name)
        except Exception:
            logger.warning(f"Band lookup failed: {band_name}, using zero fallback")
            return ee.Image.constant(0).rename(band_name)

    def add_indices(self, image: ee.Image) -> ee.Image:
        try:
            blue = self._select_band(image, "BLUE")
            green = self._select_band(image, "GREEN")
            red = self._select_band(image, "RED")
            red_edge = self._select_band(image, "RED_EDGE_1")
            nir = self._select_band(image, "NIR")
            swir = self._select_band(image, "SWIR_1")
            
            #Formulas for vegetation indices
            ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
            evi = nir.subtract(red).multiply(2.5).divide(
                nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
            ).rename("EVI")
            savi = nir.subtract(red).divide(
                nir.add(red).add(0.5)
            ).multiply(1.5).rename("SAVI")
            ndre = nir.subtract(red_edge).divide(
                nir.add(red_edge)
            ).rename("NDRE")
            ndwi = green.subtract(nir).divide(
                green.add(nir)
            ).rename("NDWI")
            ndmi = nir.subtract(swir).divide(
                nir.add(swir)
            ).rename("NDMI")

            return ee.Image.cat([ndvi, evi, savi, ndre, ndwi, ndmi])

        except Exception as e:
            logger.error(f"Error calculating vegetation indices: {e}")
            raise

    def get_all_indices(self):
        return ["NDVI", "EVI", "SAVI", "NDRE", "NDWI", "NDMI"]

    def calculate_for_image_collection(self, collection: ee.ImageCollection):
        return collection.map(self.add_indices)

    def aggregate_statistics(self, indices_collection, geometry, scale=10):
        try:
            mean_img = indices_collection.mean()

            reducers = (
                ee.Reducer.mean()
                .combine(ee.Reducer.max(), sharedInputs=True)
                .combine(ee.Reducer.min(), sharedInputs=True)
                .combine(ee.Reducer.stdDev(), sharedInputs=True)
            )

            stats = mean_img.reduceRegion(
                reducer=reducers,
                geometry=geometry,
                scale=scale,
                bestEffort=True,
                maxPixels=1e9,
            )

            result = stats.getInfo() or {}
            return result

        except Exception as e:
            logger.error(f"Error aggregating statistics: {e}")
            return {}
