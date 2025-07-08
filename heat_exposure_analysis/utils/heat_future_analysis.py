import logging
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterstats as rstat
from shapely.geometry import Point
from django.conf import settings

logger = logging.getLogger(__name__)


def _load_tiff_files(tiff_dir: Path) -> list[Path]:
    pattern = "PH_DaysOver35degC_ANN_*_20[2-4][16]-20[2-5][05].tif"
    files = sorted(tiff_dir.glob(pattern))
    if not files:
        logger.warning("No future heat GeoTIFFs found in %s", tiff_dir)
    return files


def _build_columns(files: Iterable[Path]) -> list[str]:
    timeframes = sorted({fp.stem[-9:] for fp in files})
    cols: list[str] = []
    for tf in timeframes:
        short_tf = tf[2:4] + tf[-2:]
        if short_tf == "2125":
            cols.append(f"DaysOver35C_base_{short_tf}")
        else:
            cols.append(f"DaysOver35C_ssp245_{short_tf}")
            cols.append(f"DaysOver35C_ssp585_{short_tf}")
    return cols


def generate_heat_future_analysis(df: pd.DataFrame, tiff_dir: str | Path | None = None) -> pd.DataFrame:
    """Calculate future heat exposure statistics for each facility.

    Parameters
    ----------
    df : DataFrame
        Input dataframe with ``Facility``, ``Lat`` and ``Long`` columns.
    tiff_dir : str or Path, optional
        Directory containing the future heat GeoTIFF files. Defaults to the
        ``climate_hazards_analysis/static/input_files`` directory.

    Returns
    -------
    DataFrame
        The input dataframe with additional future heat columns appended.
    """
    if tiff_dir is None:
        tiff_dir = Path(settings.BASE_DIR) / "climate_hazards_analysis" / "static" / "input_files"
    else:
        tiff_dir = Path(tiff_dir)

    files = _load_tiff_files(tiff_dir)
    if not files:
        return df

    cols = _build_columns(files)
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(xy) for xy in zip(df["Long"], df["Lat"])],
        crs="EPSG:4326",
    ).to_crs(epsg=32651)

    for col, fp in zip(cols, files):
        try:
            stats = rstat.zonal_stats(
                gdf.to_crs(epsg=4326),
                str(fp),
                stats="percentile_75",
                all_touched=True,
                geojson_out=True,
            )
            ids = [int(f["id"]) for f in stats]
            vals = [f["properties"]["percentile_75"] for f in stats]
            gdf.loc[ids, col] = vals
        except Exception as exc:
            logger.warning("Zonal stats failed for %s: %s", fp, exc)

    mask = gdf[cols].isna().any(axis=1)
    if mask.any():
        buf = gdf.loc[mask, "geometry"].buffer(1000, cap_style=3).to_crs(epsg=4326)
        for col, fp in zip(cols, files):
            try:
                stats = rstat.zonal_stats(
                    buf,
                    str(fp),
                    stats="percentile_75",
                    all_touched=True,
                    geojson_out=True,
                )
                ids = [int(f["id"]) for f in stats]
                vals = [f["properties"]["percentile_75"] for f in stats]
                gdf.loc[mask, col] = vals
            except Exception as exc:
                logger.warning("Buffered zonal stats failed for %s: %s", fp, exc)

    for col in cols:
        gdf[col] = np.ceil(gdf[col].astype(float)).astype(int)

    return gdf.drop(columns="geometry")


def apply_heat_future_analysis_to_csv(input_csv: str, tiff_dir: str | Path | None = None, output_csv: str | None = None) -> str:
    """Apply :func:`generate_heat_future_analysis` to a CSV file."""
    in_path = Path(input_csv)
    df = pd.read_csv(in_path)
    df = generate_heat_future_analysis(df, tiff_dir)
    if output_csv is None:
        output_csv = str(in_path.with_name(f"{in_path.stem}_heat_future.csv"))
    df.to_csv(output_csv, index=False)
    return output_csv