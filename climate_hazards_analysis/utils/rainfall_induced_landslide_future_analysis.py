import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterstats as rstat
from django.conf import settings

logger = logging.getLogger(__name__)

DEFAULT_TIFS = {
    "moderate": Path(
        settings.BASE_DIR,
        "climate_hazards_analysis",
        "static",
        "input_files",
        "PH_LandslideHazards_RCP26_UTM_ProjectNOAH-GIRI_Unmasked.tif",
    ),
    "worst": Path(
        settings.BASE_DIR,
        "climate_hazards_analysis",
        "static",
        "input_files",
        "PH_LandslideHazards_RCP85_UTM_ProjectNOAH-GIRI_Unmasked.tif",
    ),
}


def generate_rainfall_induced_landslide_future_analysis(
    df: pd.DataFrame,
    tif_moderate: Optional[str | Path] = None,
    tif_worst: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Append future rainfall-induced landslide values to a dataframe.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing ``Lat`` and ``Long`` columns.
    tif_moderate : str or Path, optional
        Path to the RCP26 (moderate) raster. If not provided, the default
        path within ``climate_hazards_analysis/static/input_files`` is used.
    tif_worst : str or Path, optional
        Path to the RCP85 (worst) raster. If not provided, the default
        path within ``climate_hazards_analysis/static/input_files`` is used.
    """
    tif_moderate = Path(tif_moderate) if tif_moderate else DEFAULT_TIFS["moderate"]
    tif_worst = Path(tif_worst) if tif_worst else DEFAULT_TIFS["worst"]

    tifs = {
        "Rainfall-Induced Landslide (factor of safety) - Moderate Case": tif_moderate,
        "Rainfall-Induced Landslide (factor of safety) - Worst Case": tif_worst,
    }

    for name, tif in tifs.items():
        if not tif.exists():
            logger.warning("Future landslide raster not found: %s", tif)
            df[name] = pd.NA
            continue

        gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df["Long"], df["Lat"]),
            crs="EPSG:4326",
        ).to_crs(epsg=32651)

        try:
            stats = rstat.zonal_stats(gdf, str(tif), stats="percentile_90", nodata=255)
            vals = pd.DataFrame(stats)["percentile_90"].fillna(0)
            df[name] = np.ceil(vals).astype(int)
        except Exception as exc:
            logger.warning("Zonal stats failed for %s: %s", tif, exc)
            df[name] = pd.NA

    return df


def apply_rainfall_induced_landslide_future_to_csv(
    input_csv: str,
    tif_moderate: Optional[str | Path] = None,
    tif_worst: Optional[str | Path] = None,
    output_csv: Optional[str] = None,
) -> str:
    """Apply :func:`generate_rainfall_induced_landslide_future_analysis` to a CSV file."""
    in_path = Path(input_csv)
    df = pd.read_csv(in_path)
    df = generate_rainfall_induced_landslide_future_analysis(df, tif_moderate, tif_worst)
    if output_csv is None:
        output_csv = str(in_path.with_name(f"{in_path.stem}_ril_future.csv"))
    df.to_csv(output_csv, index=False)
    return output_csv