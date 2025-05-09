import os
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
import math
from django.conf import settings

def generate_water_stress_analysis(facility_csv_path):
    """
    Performs water stress analysis for facility locations.
    
    Args:
        facility_csv_path (str): Path to the facility CSV file with locations
        
    Returns:
        dict: Dictionary containing file paths to generated outputs
    """
    try:
        # Lists to track generated files
        output_csv_files = []
        output_png_files = []
        
        # Define required input files
        water_dir = os.path.join(settings.BASE_DIR, 'water_stress', 'static', 'input_files')
        
        # Define path for climate_hazards_analysis input_files directory
        output_dir = os.path.join(settings.BASE_DIR, 'climate_hazards_analysis', 'static', 'input_files')
        os.makedirs(output_dir, exist_ok=True)
        
        shapefile_base = os.path.join(water_dir, 'hybas_lake_au_lev06_v1c')
        shapefile_path = f"{shapefile_base}.shp"
        dbf_path = f"{shapefile_base}.dbf"
        shx_path = f"{shapefile_base}.shx"
        water_risk_csv_path = os.path.join(water_dir, 'Aqueduct40_baseline_monthly_y2023m07d05.csv')
        
        # Validate input files
        required = [shapefile_path, dbf_path, shx_path, water_risk_csv_path]
        missing = [f for f in required if not os.path.exists(f)]
        if missing:
            print(f"Warning: Missing water stress files: {', '.join(missing)}")
            
            # If files are missing, just create a basic output with placeholder values
            df_fac = pd.read_csv(facility_csv_path)
            
            # Ensure Facility, Lat, Long columns exist
            rename_map = {}
            for col in df_fac.columns:
                low = col.strip().lower()
                if low in ['facility', 'site', 'site name', 'facility name', 'facilty name']:
                    rename_map[col] = 'Facility'
            if rename_map:
                df_fac.rename(columns=rename_map, inplace=True)
                
            # Add placeholder water stress column
            df_fac['Water Stress Exposure (%)'] = np.nan
            
            # Save as CSV to climate_hazards_analysis input_files directory
            output_csv = os.path.join(output_dir, 'water_stress_analysis_output.csv')
            df_fac[['Facility', 'Lat', 'Long', 'Water Stress Exposure (%)']].to_csv(output_csv, index=False)
            output_csv_files.append(output_csv)
            
            # Create a simple plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create points for plotting
            gs = gpd.points_from_xy(df_fac['Long'], df_fac['Lat'], crs='EPSG:4326')
            gdf_points = gpd.GeoDataFrame(df_fac, geometry=gs, crs='EPSG:4326')
            
            # Plot points
            gdf_points.plot(ax=ax, color='blue', markersize=100)
            
            ax.set_title('Water Stress Analysis (Placeholder)')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            
            plot_path = os.path.join(output_dir, 'water_stress_plot.png')
            plt.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            output_png_files.append(plot_path)
            
            print(f"Water stress placeholder output saved to: {output_csv}")
            
            return {
                "combined_csv_paths": output_csv_files,
                "png_paths": output_png_files
            }
        
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
            
        # Load shapefile
        hydrobasins = gpd.read_file(shapefile_path)
        if hydrobasins.crs is None:
            hydrobasins.set_crs("EPSG:4326", inplace=True)
            
        # Load water risk data
        water_risk = pd.read_csv(water_risk_csv_path)
        if 'PFAF_ID' not in hydrobasins.columns or 'PFAF_ID' not in water_risk.columns:
            raise ValueError("'PFAF_ID' must exist in both shapefile and water risk CSV")
            
        # Merge water risk data with hydrobasins
        merged = hydrobasins.merge(water_risk, on='PFAF_ID', suffixes=('_hydro', '_risk'))
        
        # Save merged shapefile in the climate_hazards_analysis input_files directory
        shp_out = os.path.join(output_dir, 'merged_aqueduct.shp')
        merged.to_file(shp_out)
        gdf = gpd.read_file(shp_out)
        
        # Create points from facility locations
        pts = [Point(x, y) for x, y in zip(df_fac['Long'], df_fac['Lat'])]
        facility_gdf = gpd.GeoDataFrame(df_fac, geometry=pts, crs="EPSG:4326")
        
        # Create buffer around points
        buf = 0.0009  # ~100m
        facility_gdf['geometry'] = facility_gdf.geometry.apply(
            lambda p: box(p.x-buf, p.y-buf, p.x+buf, p.y+buf)
        )
        
        # Clip water risk to relevant extent & normalize values
        gdf = gdf.cx[114:130, 0:20].reset_index(drop=True)
        gdf['bws_06_raw'] = gdf['bws_06_raw'].astype(float).apply(lambda v: math.ceil(v*100))
        
        # Define fields to use for water stress analysis
        dynamic_fields = ['bws_06_raw']
        
        # Spatial join to get water stress values for each facility
        facility_join = gpd.sjoin(
            facility_gdf, gdf[['geometry'] + dynamic_fields],
            how='left', predicate='intersects'
        ).reset_index(drop=True)
        
        # Add water stress values to facility data and rename to final column name
        for f in dynamic_fields:
            df_fac[f] = facility_join[f]
        
        # Rename bws_06_raw to Water Stress Exposure (%)
        df_fac.rename(columns={'bws_06_raw': 'Water Stress Exposure (%)'}, inplace=True)
            
        # Save the results to CSV in the climate_hazards_analysis input_files directory
        output_csv = os.path.join(output_dir, 'water_stress_analysis_output.csv')
        df_fac[['Facility', 'Lat', 'Long', 'Water Stress Exposure (%)']].to_csv(output_csv, index=False)
        output_csv_files.append(output_csv)
        
        # Generate a plot
        fig, ax = plt.subplots(figsize=(12, 8))
        gdf.boundary.plot(ax=ax, linewidth=1, color='black')
        gdf.plot(column='bws_06_raw', ax=ax, legend=True, cmap='OrRd')
        facility_gdf.plot(ax=ax, color='blue', markersize=50, alpha=0.5)
        ax.set_title('Water Stress and Facilities')
        ax.set_axis_off()
        
        # Save plot to the climate_hazards_analysis input_files directory
        plot_path = os.path.join(output_dir, 'water_stress_plot.png')
        plt.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        output_png_files.append(plot_path)
        
        print(f"Water stress analysis output saved to: {output_csv}")
        
        # Return paths to the generated files
        return {
            "combined_csv_paths": output_csv_files,
            "png_paths": output_png_files
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}