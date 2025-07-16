import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterstats as rstat
from django.conf import settings

logger = logging.getLogger(__name__)


DEFAULT_TIF = Path(
    settings.BASE_DIR,
    "climate_hazards_analysis",
    "static",
    "input_files",
    "PH_StormSurge_Advisory4_Future_UTM_ProjectNOAH-GIRI_Unmasked.tif",
)


def generate_storm_surge_future_analysis(
    df: pd.DataFrame, tif_path: str | Path | None = None
) -> pd.DataFrame:
    """Append future storm surge flood depth values to a dataframe.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing ``Lat`` and ``Long`` columns.
    tif_path : str or Path, optional
        Path to the future storm surge GeoTIFF. If not provided, the
        default path within ``climate_hazards_analysis/static/input_files``
        is used.

    Returns
    -------
    DataFrame
        Input dataframe with an additional
        ``Storm Surge Flood Depth (meters) - Worst Case`` column.
    """
    if tif_path is None:
        tif = DEFAULT_TIF
    else:
        tif = Path(tif_path)

    if not tif.exists():
        logger.warning("Future storm surge raster not found: %s", tif)
        df["Storm Surge Flood Depth (meters) - Worst Case"] = pd.NA
        return df

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["Long"], df["Lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=32651)

    stats = rstat.zonal_stats(gdf, str(tif), stats="max", nodata=255)
    vals = pd.DataFrame(stats)["max"].fillna(0).astype(int)
    gdf["Storm Surge Flood Depth (meters) - Worst Case"] = vals

    return gdf.drop(columns="geometry")


def apply_storm_surge_future_to_csv(
    input_csv: str, tif_path: str | Path | None = None, output_csv: str | None = None
) -> str:
    """Apply :func:`generate_storm_surge_future_analysis` to a CSV file."""
    in_path = Path(input_csv)
    df = pd.read_csv(in_path)
    df = generate_storm_surge_future_analysis(df, tif_path)
    if output_csv is None:
        output_csv = str(in_path.with_name(f"{in_path.stem}_storm_surge_future.csv"))
    df.to_csv(output_csv, index=False)
    return output_csv