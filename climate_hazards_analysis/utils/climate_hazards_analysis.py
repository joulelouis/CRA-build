import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
from django.conf import settings
from rasterstats import zonal_stats
from pyproj import CRS

def generate_climate_hazards_analysis(shapefile_path, dbf_path, shx_path, water_risk_csv_path,
                               facility_csv_path, raster_path,
                               water_dynamic_fields=None, water_plot_fields=None):
    """
    Integrates water stress analysis and flood exposure analysis.
    
    Water stress analysis:
      - Loads a shapefile and a water risk CSV.
      - Merges them on 'PFAF_ID' and spatially joins water risk fields (e.g. bws_06_lab)
        with facility locations from facility_csv_path.
      - Generates a water stress plot.
      
    Flood exposure analysis:
      - Uses the updated facility data (with water stress field) to compute exposure
        from a raster (via zonal stats) and adds an Exposure field.
      
    Finally, the function writes a CSV table with the columns:
      Site, Lat, Long, bws_06_lab, Exposure
    and returns the path to the water stress plot along with the combined CSV path.
    """
    try:
        # ---------- WATER STRESS ANALYSIS ----------
        print("Step 1: Validating input files for water stress analysis")
        # Check shapefile components
        required_files = [shapefile_path, dbf_path, shx_path]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"Missing shapefile components: {', '.join(missing_files)}")
        if not os.path.exists(water_risk_csv_path):
            raise FileNotFoundError(f"Water risk CSV {water_risk_csv_path} not found!")
        if not os.path.exists(facility_csv_path):
            raise FileNotFoundError(f"Facility CSV {facility_csv_path} not found!")
        
        print("Step 2: Loading shapefile")
        hydrobasins = gpd.read_file(shapefile_path)
        if hydrobasins.crs is None:
            print("WARNING: Shapefile has no CRS! Setting to EPSG:4326...")
            hydrobasins.set_crs("EPSG:4326", inplace=True)
        print(f"Shapefile CRS: {hydrobasins.crs}")
        
        print("Step 3: Loading water risk CSV")
        water_risk_data = pd.read_csv(water_risk_csv_path)
        
        # Ensure merging key exists
        if 'PFAF_ID' not in hydrobasins.columns:
            raise ValueError("PFAF_ID not found in shapefile")
        if 'PFAF_ID' not in water_risk_data.columns:
            raise ValueError("PFAF_ID not found in water risk CSV")
        
        print("Step 4: Merging water risk data with shapefile")
        merged_data = hydrobasins.merge(water_risk_data, on='PFAF_ID', suffixes=('_hydro', '_risk'))
        
        # Save merged shapefile to a temporary directory
        ws_uploads_dir = os.path.join(settings.BASE_DIR, 'water_stress', 'static', 'input_files')
        os.makedirs(ws_uploads_dir, exist_ok=True)
        merged_shp_path = os.path.join(ws_uploads_dir, 'merged_aqueduct.shp')
        merged_data.to_file(merged_shp_path)
        print(f"Merged shapefile saved at {merged_shp_path}")
        
        # Reload merged shapefile for further processing
        gdf = gpd.read_file(merged_shp_path)
        
        # ---------- FACILITY DATA PREPARATION ----------
        print("Step 5: Loading and cleaning facility CSV")
        facility_locs = pd.read_csv(facility_csv_path)
        for col in ['Long', 'Lat']:
            if col not in facility_locs.columns:
                raise ValueError(f"Missing '{col}' column in facility CSV.")
        facility_locs['Long'] = pd.to_numeric(facility_locs['Long'], errors='coerce')
        facility_locs['Lat'] = pd.to_numeric(facility_locs['Lat'], errors='coerce')
        facility_locs = facility_locs.dropna(subset=['Long', 'Lat'])
        if facility_locs.empty:
            raise ValueError("Facility CSV is empty after cleaning!")
        
        # Convert to GeoDataFrame using facility coordinates
        geometry = [Point(xy) for xy in zip(facility_locs['Long'], facility_locs['Lat'])]
        facility_gdf = gpd.GeoDataFrame(facility_locs, geometry=geometry, crs="EPSG:4326")
        
        # Create small square buffers (approx ~100m) for spatial join
        buffer_size = 0.0009
        facility_gdf['geometry'] = facility_gdf.geometry.apply(
            lambda pt: box(pt.x - buffer_size, pt.y - buffer_size, pt.x + buffer_size, pt.y + buffer_size)
        )
        
        # Filter the merged gdf to a region (if needed)
        min_lat, max_lat = 0, 20
        min_lon, max_lon = 114, 130
        gdf = gdf.cx[min_lon:max_lon, min_lat:max_lat]
        
        # Reset index before spatial join
        gdf = gdf.reset_index(drop=True)
        print("Step 6: Performing spatial join to add water stress data")
        # Set default dynamic fields if not provided (we need at least bws_06_lab)
        if water_dynamic_fields is None:
            water_dynamic_fields = ['bws_06_lab']
        for field in water_dynamic_fields:
            if field not in gdf.columns:
                raise ValueError(f"Field {field} not found in water risk data.")
        facility_gdf = gpd.sjoin(facility_gdf, gdf[['geometry'] + water_dynamic_fields],
                                  how='left', predicate='intersects')
        facility_gdf = facility_gdf.reset_index(drop=True)
        
        # Update facility_locs with the joined dynamic field(s)
        for field in water_dynamic_fields:
            facility_locs[field] = facility_gdf[field]
        
        # Save the updated facility CSV (water stress part)
        updated_facility_csv = os.path.join(ws_uploads_dir, 'combined_facility_ws.csv')
        facility_locs.to_csv(updated_facility_csv, index=False)
        print(f"Updated facility CSV with water stress data saved at {updated_facility_csv}")
        
        # ---------- WATER STRESS PLOT ----------
        print("Step 7: Generating water stress plot")
        fig, ax = plt.subplots(figsize=(12, 8))
        gdf.boundary.plot(ax=ax, linewidth=1, color='black')
        # Optionally plot each water stress field; here we use the first one
        for field in (water_plot_fields or water_dynamic_fields):
            if field in gdf.columns:
                gdf.plot(column=field, ax=ax, legend=True, cmap='OrRd')
            else:
                print(f"Warning: {field} not found in merged data for plotting")
        # Plot facility locations
        facility_gdf.plot(ax=ax, color='blue', marker='o', markersize=50, alpha=0.5)
        ax.set_title('Water Stress and Facility Locations', fontsize=15)
        ax.set_axis_off()
        plot_path = os.path.join(ws_uploads_dir, 'water_stress_plot.png')
        plt.savefig(plot_path, format='png')
        plt.close(fig)
        print(f"Water stress plot saved at {plot_path}")
        
        # ---------- FLOOD EXPOSURE ANALYSIS ----------
        print("Step 8: Performing flood exposure analysis")
        # For flood exposure, we use the updated facility data saved above.
        # Re-load facility CSV (it must have at least Site, Lat, Long)
        df_fac = pd.read_csv(updated_facility_csv)
        # Ensure required columns exist
        for col in ['Long', 'Lat', 'Site']:
            if col not in df_fac.columns:
                raise ValueError(f"Missing '{col}' column in facility CSV for flood analysis.")
        
        # Set a larger buffer (~250 meters radius)
        flood_buffer = 0.0045
        crs_flood = CRS("epsg:32651")  # example projection; adjust as needed
        
        # Create geometries (using a different buffer for flood analysis)
        geometry = [Point(xy).buffer(flood_buffer, cap_style=3) for xy in zip(df_fac['Long'], df_fac['Lat'])]
        flood_gdf = gpd.GeoDataFrame(df_fac, crs=crs_flood, geometry=geometry)
        
        print("Computing zonal statistics on flood raster")
        stats = zonal_stats(flood_gdf.geometry, raster_path, stats="percentile_75")
        # Add 75th Percentile values to GeoDataFrame (default to 1 if missing)
        flood_gdf['75th Percentile'] = [stat.get('percentile_75', 1) if stat.get('percentile_75') is not None else 1 for stat in stats]
        
        # Define Exposure based on the 75th Percentile value
        def determine_exposure(val):
            if val == 1:
                return 'Low'
            elif val == 2:
                return 'Medium'
            elif val == 3:
                return 'High'
            else:
                return 'Unknown'
        flood_gdf['Exposure'] = flood_gdf['75th Percentile'].apply(determine_exposure)
        
        # ---------- GENERATE COMBINED OUTPUT TABLE ----------
        print("Step 9: Generating combined output table")
        # We require fields: Site, Lat, Long, bws_06_lab, Exposure
        required_columns = ['Site', 'Lat', 'Long', 'bws_06_lab', 'Exposure']
        for col in required_columns:
            if col not in flood_gdf.columns:
                raise ValueError(f"Expected column '{col}' not found in combined data.")
        
        combined_df = flood_gdf[required_columns]
        combined_uploads_dir = os.path.join(settings.BASE_DIR, 'climate_hazards_analysis', 'static', 'input_files')
        os.makedirs(combined_uploads_dir, exist_ok=True)
        combined_csv_path = os.path.join(combined_uploads_dir, 'combined_output.csv')
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Combined output CSV saved at {combined_csv_path}")
        
        return {
            'plot_path': plot_path,
            'combined_csv_path': combined_csv_path
        }
        
    except Exception as e:
        print(f"Error in generate_combined_analysis: {e}")
        return None
