import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from pyproj import CRS
from rasterstats import zonal_stats
from django.conf import settings

def generate_flood_exposure_analysis(facility_csv_path, raster_path):
    try:
        # Set square buffer around point location in the absence of facility extent
        buffer = 0.0045  # degree, ~250 meters radius, ~500 m x 500 m

        # Set geospatial map projection
        proj1 = 4326

        # Set folder paths and input files
        folder = Path(r"D:\Documents\NOAH\Flood")
        out_folder = r"D:\Documents\NOAH\Flood\Trial\Exposure"
        # facility_csv_path = "sample_locs.csv"
        # raster_path = "Abra_Flood_100year.tif"

        # Open facility location and georeference
        df_fac = pd.read_csv(facility_csv_path)
        crs = CRS("epsg:32651")

        # Create geometry from Long and Lat
        geometry = [Point(xy).buffer(buffer, cap_style=3) for xy in zip(df_fac['Long'], df_fac['Lat'])]
        geo_df = gpd.GeoDataFrame(df_fac, crs=crs, geometry=geometry)

        # Zonal statistics
        stats = zonal_stats(geo_df.geometry, raster_path, stats="percentile_75")
        geo_df['75th Percentile'] = [stat['percentile_75'] if stat['percentile_75'] is not None else 1 for stat in stats]

        # Define Exposure based on 75th Percentile
        def determine_exposure(percentile):
            if percentile == 1:
                return 'Low'
            elif percentile == 2:
                return 'Medium'
            elif percentile == 3:
                return 'High'
            else:
                return 'Unknown'

        geo_df['Exposure'] = geo_df['75th Percentile'].apply(determine_exposure)

        
        uploads_dir = os.path.join(settings.BASE_DIR, 'flood_exposure_analysis', 'static', 'input_files')
        os.makedirs(uploads_dir, exist_ok=True)

        # Save to CSV
        output_csv_path = os.path.join(uploads_dir, 'output_with_exposure.csv')
        geo_df[['Site', 'Long', 'Lat', '75th Percentile', 'Exposure']].to_csv(output_csv_path, index=False)
        print("CSV file saved successfully.")

    except Exception as e:
        print(f"Error in generate_flood_exposure_analysis: {e}")
        return None