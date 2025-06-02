import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from pyproj import CRS
from rasterstats import zonal_stats
from django.conf import settings

def generate_flood_exposure_analysis(facility_csv_path, buffer_size=0.0045):
    """
    Performs flood exposure analysis for facility locations.
    Args:
    facility_csv_path (str): Path to the facility CSV file with locations
    buffer_size (float): Buffer size for analysis (~250m = 0.0045, ~500m = 0.0090, ~750m = 0.0135)
    
    Returns:
        dict: Dictionary containing file paths to generated outputs
    """
    try:
        # Lists to track generated files
        output_csv_files = []
        output_png_files = []
        
        # Define path for climate_hazards_analysis input_files directory
        output_dir = os.path.join(settings.BASE_DIR, 'climate_hazards_analysis', 'static', 'input_files')
        os.makedirs(output_dir, exist_ok=True)
        
        # Path to the flood raster file
        raster_path = os.path.join(
            settings.BASE_DIR, 'flood_exposure_analysis', 'static', 
            'input_files', 'Abra_Flood_100year.tif'
        )
        
        # Check if raster file exists
        if not os.path.exists(raster_path):
            print(f"Warning: Flood raster file not found: {raster_path}")
            raster_path = os.path.join(
                settings.BASE_DIR, 'climate_hazards_analysis', 'static', 
                'input_files', 'Abra_Flood_100year.tif'
            )
            if not os.path.exists(raster_path):
                raise FileNotFoundError("Flood raster file not found in either flood_exposure_analysis or climate_hazards_analysis directories")
        
        # Load facility locations
        df_fac = pd.read_csv(facility_csv_path)
        
        # Ensure Facility, Lat, Long columns exist
        rename_map = {}
        for col in df_fac.columns:
            low = col.strip().lower()
            if low in ['facility', 'site', 'site name', 'facility name', 'facilty name']:
                rename_map[col] = 'Facility'
        if rename_map:
            df_fac.rename(columns=rename_map, inplace=True)
            
        # Ensure coordinates are present
        for coord in ['Long', 'Lat']:
            if coord not in df_fac.columns:
                raise ValueError(f"Missing '{coord}' column in facility CSV.")
        
        # Convert to numeric and drop invalid coordinates
        df_fac['Long'] = pd.to_numeric(df_fac['Long'], errors='coerce')
        df_fac['Lat'] = pd.to_numeric(df_fac['Lat'], errors='coerce')
        df_fac.dropna(subset=['Long', 'Lat'], inplace=True)
        
        if 'Facility' not in df_fac.columns:
            raise ValueError("Your facility CSV must include a 'Facility' column or equivalent header.")
        
        # Use the provided buffer size (default ~250 meters for 0.0045)
        buffer = buffer_size
        buffer_meters = int(buffer * 111000)  # Convert to approximate meters for labeling
        
        print(f"Using buffer size: {buffer} degrees (~{buffer_meters}m)")
        
        # Create polygons with buffer for flood analysis
        flood_polys = [
            Point(x, y).buffer(buffer, cap_style=3) 
            for x, y in zip(df_fac['Long'], df_fac['Lat'])
        ]
        
        # Create GeoDataFrame with proper projection
        flood_gdf = gpd.GeoDataFrame(df_fac.copy(), geometry=flood_polys, crs=CRS('epsg:32651'))
        
        # Extract 75th percentile flood depth from raster
        stats = zonal_stats(flood_gdf.geometry, raster_path, stats='percentile_75')
        
        # Process the results
        flood_gdf['75th Percentile'] = [
            stat['percentile_75'] if stat['percentile_75'] is not None else 1 
            for stat in stats
        ]
        
        # Classify exposure based on flood depth
        def determine_exposure(percentile):
            if pd.isna(percentile):
                return 'Unknown'
            if percentile == 1:
                return '0.1 to 0.5'
            elif percentile == 2:
                return '0.5 to 1.5'
            else:
                return 'Greater than 1.5'
        
        # Apply classification and add to GeoDataFrame
        flood_gdf['Exposure'] = flood_gdf['75th Percentile'].apply(determine_exposure)
        
        # Rename 'Exposure' to 'Flood Depth (meters)' for the correct column name
        flood_gdf.rename(columns={'Exposure': 'Flood Depth (meters)'}, inplace=True)
        
        # Create a visualization of the flood exposure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a colormap for different exposure categories
        cmap = {
            '0.1 to 0.5': 'green',
            '0.5 to 1.5': 'orange',
            'Greater than 1.5': 'red',
            'Unknown': 'gray'
        }
        
        # Create a point-based GeoDataFrame for visualization
        points_gdf = gpd.GeoDataFrame(
            flood_gdf[['Facility', 'Lat', 'Long', 'Flood Depth (meters)']],
            geometry=gpd.points_from_xy(flood_gdf['Long'], flood_gdf['Lat']),
            crs='EPSG:4326'
        )
        
        # Simple plot with facility locations
        points_gdf.plot(ax=ax, color='blue', markersize=100)
        
        ax.set_title(f'Flood Exposure for Facility Locations (Buffer: ~{buffer_meters}m)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Save the plot with buffer size in filename for sensitivity analysis
        plot_filename = f'flood_exposure_plot_buffer_{buffer:.4f}.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        output_png_files.append(plot_path)
        
        # Save the results to CSV with buffer size in filename for sensitivity analysis
        output_filename = f'flood_exposure_analysis_output_buffer_{buffer:.4f}.csv'
        output_csv = os.path.join(output_dir, output_filename)
        
        # Only include relevant columns in the output
        flood_gdf[['Facility', 'Lat', 'Long', 'Flood Depth (meters)']].to_csv(
            output_csv, index=False
        )
        output_csv_files.append(output_csv)
        
        print(f"Flood analysis output saved to: {output_csv}")
        
        # Return paths to the generated files
        return {
            "combined_csv_paths": output_csv_files,
            "png_paths": output_png_files,
            "buffer_size": buffer_size,
            "buffer_meters": buffer_meters
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}            