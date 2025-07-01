import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterstats as rstat
from django.conf import settings

logger = logging.getLogger(__name__)


def generate_heat_future_analysis(df):
    """Add future heat exposure values for SSP245 and SSP585 scenarios.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing ``Lat`` and ``Long`` columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with additional columns for the 75th percentile of days
        over 35°C for three future timeframes under SSP245 and SSP585.
    """
    try:
        idir = (
            Path(settings.BASE_DIR)
            / "climate_hazards_analysis"
            / "static"
            / "input_files"
        )

        grid_dir = idir
        fps = sorted(
            grid_dir.glob(
                "PH_DaysOver35degC_ANN_*_20[2-4][1|6]-20[2-5][0|5].tif"
            )
        )

        if not fps:
            logger.warning("No matching GeoTIFFs found in %s", grid_dir)
            return df

        timeframes = sorted({fp.stem[-9:] for fp in fps})
        cols_35 = []
        for tf in timeframes:
            short_tf = tf[2:4] + tf[-2:]
            if short_tf == "2125":
                cols_35.append(f"DaysOver35C_base_{short_tf}")
            else:
                cols_35.append(f"DaysOver35C_ssp245_{short_tf}")
                cols_35.append(f"DaysOver35C_ssp585_{short_tf}")

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["Long"], df["Lat"]),
            crs="EPSG:4326",
        ).to_crs(epsg=32651)

        for col, fp in zip(cols_35, fps):
            gdf[col] = np.nan
       
            stats = rstat.zonal_stats(
                gdf.to_crs(epsg=4326),
                str(fp),
                stats="percentile_75",
                all_touched=True,
                geojson_out=True,
            )
            ids = [int(feat["id"]) for feat in stats]
            vals = [feat["properties"]["percentile_75"] for feat in stats]
            gdf.loc[ids, col] = vals

        mask = gdf[cols_35].isna().any(axis=1)
        if mask.any():
            buf = gdf.loc[mask].geometry.buffer(1000, cap_style=3).to_crs(epsg=4326)
            for col, fp in zip(cols_35, fps):
                stats = rstat.zonal_stats(
                    buf,
                    str(fp),
                    stats="percentile_75",
                    all_touched=True,
                    geojson_out=True,
                )
                ids = [int(feat["id"]) for feat in stats]
                vals = [feat["properties"]["percentile_75"] for feat in stats]
                gdf.loc[mask, col] = vals

        for col in cols_35:
            gdf[col] = gdf[col].apply(
                lambda v: int(np.ceil(float(v))) if pd.notnull(v) else np.nan
            )

        try:
            df_out = gdf.drop(columns=["geometry"])
            df_out.to_excel(idir / "HeatExposure_future_all.xlsx", index=False)
        except Exception as exc:
            logger.warning("Could not export future heat Excel file: %s", exc)

        df[cols_35] = gdf[cols_35]

    except Exception as e:
        logger.exception("Error in generate_heat_future_analysis: %s", e)

    return df