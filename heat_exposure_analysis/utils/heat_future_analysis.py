import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterstats as rstat
from django.conf import settings

logger = logging.getLogger(__name__)


def generate_heat_future_analysis(df):
    """Add future heat exposure values for SSP585 scenarios.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing ``Lat`` and ``Long`` columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with additional columns for the 75th percentile of days
        over 35°C for three future timeframes.
    """
    try:
        idir = (
            Path(settings.BASE_DIR)
            / "climate_hazards_analysis"
            / "static"
            / "input_files"
        )

        timeframe_map = {
            "2630": "2026-2030",
            "3140": "2031-2040",
            "4150": "2041-2050",
        }

        # Build mapping of timeframe -> raster path (may be missing)
        fp_map = {}
        for tf, span in timeframe_map.items():
            fp = idir / f"PH_DaysOver35degC_ANN_ssp585_{span}.tif"
            if fp.exists():
                fp_map[tf] = fp
            else:
                logger.warning("Future heat raster not found: %s", fp)

        # Always create the output columns even if rasters are missing
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["Long"], df["Lat"]),
            crs="EPSG:4326",
        )

        temp_cols = [f"n>35degC_ssp585_{tf}" for tf in timeframe_map.keys()]
        for col in temp_cols:
            gdf[col] = np.nan
        geojson_all = gdf.__geo_interface__
        for tf, fp in fp_map.items():
            col = f"n>35degC_ssp585_{tf}"
            stats = rstat.zonal_stats(
                geojson_all,
                str(fp),
                stats="percentile_75",
                all_touched=True,
                geojson_out=True,
            )
            ids = [int(feat["id"]) for feat in stats]
            vals = [feat["properties"]["percentile_75"] for feat in stats]
            gdf[col] = pd.Series(vals, index=ids)

        missing = gdf[[c for c in temp_cols if c in gdf.columns]].isna().any(axis=1)
        if missing.any() and fp_map:
            buffered = (
                gdf.loc[missing]
                .to_crs(epsg=32651)
                .geometry.buffer(1000, cap_style="square", join_style="mitre")
                .to_crs("EPSG:4326")
            )
            geojson_buf = gpd.GeoSeries(buffered, crs="EPSG:4326").__geo_interface__
            for tf, fp in fp_map.items():
                col = f"n>35degC_ssp585_{tf}"
                stats = rstat.zonal_stats(
                    geojson_buf,
                    str(fp),
                    stats="percentile_75",
                    all_touched=True,
                    geojson_out=True,
                )
                ids = [int(feat["id"]) for feat in stats]
                vals = [feat["properties"]["percentile_75"] for feat in stats]
                gdf.loc[missing, col] = pd.Series(vals, index=ids)

        for col in temp_cols:
            gdf[col] = gdf[col].apply(lambda v: int(np.ceil(v)) if pd.notnull(v) else np.nan)

        try:
            df_out = gdf.drop(columns=["geometry"])
            df_out.to_excel(idir / "HeatExposure_ssp585_AllTimeframes.xlsx", index=False)
        except Exception as exc:  # e.g., openpyxl not installed
            logger.warning("Could not export future heat Excel file: %s", exc)

        df[temp_cols] = gdf[temp_cols]
    except Exception as e:
        logger.exception("Error in generate_heat_future_analysis: %s", e)

    return df