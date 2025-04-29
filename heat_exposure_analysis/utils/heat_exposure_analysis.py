import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
import rasterstats as rstat
import math
from django.conf import settings

def generate_heat_exposure_analysis(facility_csv_path):
    """
    Performs heat exposure analysis for facility locations.
    
    Args:
        facility_csv_path (str): Path to the facility CSV file with locations
        
    Returns:
        dict: Dictionary containing file paths to generated outputs
    """
    try:
        # Lists to track generated files
        output_csv_files = []
        output_png_files = []
        
        # Ensure input directories exist
        idir = Path(settings.BASE_DIR)/'climate_hazards_analysis'/'static'/'input_files'
        os.makedirs(idir, exist_ok=True)
        
        # Define required heat exposure raster files
        heat_files = [
            idir/'PH_DaysOver30degC_ANN_2021-2025.tif',
            idir/'PH_DaysOver33degC_ANN_2021-2025.tif',
            idir/'PH_DaysOver35degC_ANN_2021-2025.tif'
        ]
        
        # Check if heat raster files exist - allow proceed even if missing
        missing = [str(f) for f in heat_files if not os.path.exists(f)]
        if missing:
            print(f"Warning: Missing heat exposure raster files: {', '.join(missing)}")
            # Continue instead of raising exception - we'll handle missing data
        
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
        
        # Define heat exposure columns
        heat_cols = [f"n>{t}degC_2125" for t in [30, 33, 35]]
        
        # Create initial DataFrame with placeholder values
        df_heat = df_fac[['Facility', 'Lat', 'Long']].copy()
        
        # Add placeholder columns for heat data (in case we can't process the rasters)
        for col in heat_cols:
            df_heat[col] = np.nan
        
        # Only try to process rasters if they exist
        existing_heat_files = [f for f in heat_files if os.path.exists(f)]
        if existing_heat_files:
            try:
                # Create points from facility locations with projection
                gs = gpd.points_from_xy(df_fac['Long'], df_fac['Lat'], crs='EPSG:4326').to_crs('EPSG:32651')
                gdf_heat = gpd.GeoDataFrame(df_fac, geometry=gs, crs='EPSG:32651')
                
                # Add buffer around each point (default lot_area if not specified)
                gdf_heat['lot_area'] = gdf_heat.get('lot_area', 1000**2)
                gdf_heat['geometry'] = gdf_heat.geometry.buffer(
                    np.sqrt(gdf_heat['lot_area'])/2, cap_style='square', join_style='mitre')
                
                # Create a temporary geojson file for rasterstats
                temp_geo = Path('temp.features.geojson')
                
                # For each heat threshold, extract values from the raster
                for i, (col, fp) in enumerate(zip(heat_cols, heat_files)):
                    if os.path.exists(fp):
                        try:
                            # Convert to WGS84 for the geojson file
                            gdf_heat.to_crs('EPSG:4326').to_file(str(temp_geo), driver='GeoJSON')
                            # Run zonal statistics
                            out = rstat.zonal_stats(str(temp_geo), str(fp), stats='percentile_75', 
                                                    all_touched=True, geojson_out=True)
                            
                            if out:  # Check if we got results
                                idxs = [int(feat['id']) for feat in out]
                                vals = [feat['properties']['percentile_75'] for feat in out]
                                gdf_heat[col] = pd.Series(vals, index=idxs)
                                
                                # Update the main DataFrame
                                df_heat[col] = gdf_heat[col]
                        except Exception as e:
                            print(f"Error processing heat raster {fp.name}: {e}")
                        finally:
                            # Clean up temp file if it exists
                            if temp_geo.exists():
                                temp_geo.unlink()
            except Exception as e:
                print(f"Error in heat raster processing: {e}")
        
        # Round and clean up heat values
        for col in heat_cols:
            if col in df_heat.columns:
                df_heat.loc[:, col] = df_heat[col].apply(
                    lambda v: int(math.ceil(v)) if pd.notnull(v) else v)
        
        # Generate a simplified plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a point-based GeoDataFrame for visualization
        point_geom = gpd.points_from_xy(df_heat['Long'], df_heat['Lat'], crs='EPSG:4326')
        gdf_points = gpd.GeoDataFrame(df_heat, geometry=point_geom, crs='EPSG:4326')
        
        # Simple plot with facility locations
        gdf_points.plot(ax=ax, color='red', markersize=100)
        
        ax.set_title('Heat Exposure Analysis for Facility Locations')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Save the plot to the climate_hazards_analysis input_files directory
        output_dir = os.path.join(settings.BASE_DIR, 'climate_hazards_analysis', 'static', 'input_files')
        os.makedirs(output_dir, exist_ok=True)
        
        plot_path = os.path.join(output_dir, 'heat_exposure_plot.png')
        plt.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        output_png_files.append(plot_path)
        
        # Save the results to CSV in the climate_hazards_analysis input_files directory
        output_csv = os.path.join(output_dir, 'heat_exposure_analysis_output.csv')
        df_heat.to_csv(output_csv, index=False)
        output_csv_files.append(output_csv)
        
        print(f"Heat analysis output saved to: {output_csv}")
        
        # Return paths to the generated files
        return {
            "combined_csv_paths": output_csv_files,
            "png_paths": output_png_files
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}