
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rioxarray as rx
from django.conf import settings
import os


def generate_sea_level_rise_analysis(facility_csv_path):
    try:
        
        output_dir = os.path.join(settings.BASE_DIR, 'sea_level_rise_analysis', 'static', 'output')
        os.makedirs(output_dir, exist_ok=True)

        df_points = pd.read_csv(facility_csv_path)


        # Standardize expected column names
        rename_map = {}
        for col in df_points.columns:
            low = col.strip().lower()
            if low in ['facility', 'site', 'site name', 'facility name', 'facilty name', 'name', 'asset name']:
                rename_map[col] = 'Facility'
            elif low in ['latitude', 'lat'] and 'Lat' not in df_points.columns:
                rename_map[col] = 'Lat'
            elif low in ['longitude', 'long', 'lon'] and 'Long' not in df_points.columns:
                rename_map[col] = 'Long'
        if rename_map:
            df_points.rename(columns=rename_map, inplace=True)

        required_cols = ['Facility', 'Lat', 'Long']
        missing = [c for c in required_cols if c not in df_points.columns]
        if missing:
            raise ValueError(f"Missing required columns in facility CSV: {', '.join(missing)}")


        geometry = [Point(xy) for xy in zip(df_points['Long'], df_points['Lat'])]
        gpd.GeoDataFrame(df_points, geometry=geometry, crs='EPSG:4326')
        df_master = df_points[['Facility', 'Lat', 'Long']].rename(columns={'Lat': 'LAT', 'Long': 'LON'})

        quant = 0.5
        year_list = np.arange(2030, 2051, 10)
        ssp_list = ['245', '585']

        for ssp in ssp_list:
            for year in year_list:
                file_path = os.path.join(settings.BASE_DIR, 'sea_level_rise_analysis', 'static', 'input_files', f'total_{year}_ssp{ssp}_Medium.nc')
                if not os.path.exists(file_path):
                    print(f'\u26a0\ufe0f Missing file: {file_path}')
                    continue
                ds_slr = xr.open_dataset(file_path, engine='netcdf4')
                ds_slr = ds_slr.sel(lon=slice(115.0, 128.0), lat=np.arange(3, 22))
                ds_slr = ds_slr.sel(CI=quant)
                ds_slr = ds_slr.where(ds_slr['sealevel_mm'] > -32768)

                col_name = f'{year}_SLR_SSP{ssp}_m'
                slr_values = []
                for row in df_master.itertuples():
                    slr_proj = round(float(ds_slr.sel(lon=row.LON, lat=row.LAT, method='nearest').sealevel_mm.values) / 1000, 2)
                    slr_values.append(slr_proj)
                df_master[col_name] = slr_values

        rename_map = {
            '2030_SLR_SSP245_m': '2030 Sea Level Rise (meters) - Moderate Case',
            '2040_SLR_SSP245_m': '2040 Sea Level Rise (meters) - Moderate Case',
            '2050_SLR_SSP245_m': '2050 Sea Level Rise (meters) - Moderate Case',
            '2030_SLR_SSP585_m': '2030 Sea Level Rise (meters) - Worst Case',
            '2040_SLR_SSP585_m': '2040 Sea Level Rise (meters) - Worst Case',
            '2050_SLR_SSP585_m': '2050 Sea Level Rise (meters) - Worst Case',
        }
        df_master.rename(columns=rename_map, inplace=True)

        output_file = os.path.join(output_dir, 'combined_slr_SSP245_SSP585_median.csv')
        df_master.to_csv(output_file, index=False)

        return {'combined_csv_paths': [output_file], 'png_paths': []}
    except Exception as e:
        return {"error": str(e)}
        