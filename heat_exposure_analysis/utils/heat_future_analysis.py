import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterstats as rstat
from shapely.geometry import Point, box
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
            grid_dir.glob("PH_DaysOver35degC_ANN_*_20[2-4][1|6]-20[2-5][0|5].tif")
        )

        if not fps:
            logger.error("No matching GeoTIFFs found in %s", grid_dir)
            return df

        logger.debug("Found %d GeoTIFF files", len(fps))

        # Ensure CRS match between input points and rasters
        with rasterio.open(fps[0]) as src:
            raster_crs = src.crs
            raster_bounds = box(*src.bounds)

        if raster_crs is None:
            logger.error("Raster %s lacks CRS information", fps[0])
            return df

        # quick check that the site locations overlap the raster extent
        bounds = gpd.GeoSeries([raster_bounds], crs=raster_crs)
        gdf = gpd.GeoDataFrame(
            df.reset_index(drop=True),
            geometry=gpd.points_from_xy(df["Long"], df["Lat"]),
            crs="EPSG:4326",
        )
        gdf_raster = gdf.to_crs(raster_crs)
        if not bounds.intersects(gdf_raster.unary_union.envelope).bool():
            logger.error(
                "Input locations do not overlap the raster extent. Check CRS or data coordinates"
            )
            return df

        timeframes = sorted({fp.stem[-9:] for fp in fps})

        # Determine expected output columns based on available timeframes
        expected_cols = []
        for tf in timeframes:
            short_tf = tf[2:4] + tf[-2:]
            if short_tf == "2125":
                expected_cols.append(f"DaysOver35C_base_{short_tf}")
            else:
                expected_cols.extend(
                    [
                        f"DaysOver35C_ssp245_{short_tf}",
                        f"DaysOver35C_ssp585_{short_tf}",
                    ]
                )

        for col in expected_cols:
            if col not in df.columns:
                df[col] = np.nan

        
        # Map each raster file to its corresponding column name in a list
        cols_35 = []
        fp_order = []
        for fp in fps:
            tf = fp.stem[-9:]
            short_tf = tf[2:4] + tf[-2:]
            if "ssp245" in fp.stem:
                cols_35.append(f"DaysOver35C_ssp245_{short_tf}")
                fp_order.append(fp)
            elif "ssp585" in fp.stem:
                cols_35.append(f"DaysOver35C_ssp585_{short_tf}")
        
                fp_order.append(fp)
            else:
                cols_35.append(f"DaysOver35C_base_{short_tf}")
                fp_order.append(fp)

        gdf_metric = gdf.to_crs(epsg=32651)

        if gdf.crs != raster_crs:
            logger.warning(
                "Reprojecting points from %s to %s to match rasters",
                gdf.crs,
                raster_crs,
            )
        gdf = gdf_raster

        for col, fp in zip(cols_35, fp_order):
            stats = rstat.zonal_stats(
                gdf,
                str(fp),
                stats="percentile_75",
                all_touched=True,
                geojson_out=True,
            )
            if stats:
                ids = [int(feat["id"]) for feat in stats]
                vals = [feat["properties"]["percentile_75"] for feat in stats]
                gdf.loc[ids, col] = vals

        mask = gdf[cols_35].isna().any(axis=1)
        if mask.any():
            buf_metric = gdf_metric.loc[mask].geometry.buffer(1000, cap_style=3)
            buf = gpd.GeoSeries(buf_metric, crs=gdf_metric.crs).to_crs(raster_crs)
            for col, fp in zip(cols_35, fp_order):
                stats = rstat.zonal_stats(
                    buf,
                    str(fp),
                    stats="percentile_75",
                    all_touched=True,
                    geojson_out=True,
                )
                if stats:
                    vals = [feat["properties"]["percentile_75"] for feat in stats]
                    
                    gdf.loc[mask, col] = vals

        for col in cols_35:
            gdf[col] = gdf[col].apply(
                lambda v: int(np.ceil(float(v))) if pd.notnull(v) else np.nan
            )

        missing = int(gdf[cols_35].isna().sum().sum())
        if missing:
            logger.warning(
                "%d future heat values remain missing after processing", missing
            )

        try:
            df_out = gdf.drop(columns=["geometry"])
            df_out.to_excel(idir / "HeatExposure_future_all.xlsx", index=False)
        except Exception as exc:
            logger.warning("Could not export future heat Excel file: %s", exc)

        
        df[expected_cols] = gdf[expected_cols]

    except Exception as e:
        logger.exception("Error in generate_heat_future_analysis: %s", e)

    return df