import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
from django.conf import settings
from rasterstats import zonal_stats
from pyproj import CRS

# Import the sea level rise analysis function from its module.
from sea_level_rise_analysis.utils.sea_level_rise_analysis import generate_sea_level_rise_analysis
# Import the tropical cyclone analysis function from its module.
from tropical_cyclone_analysis.utils.tropical_cyclone_analysis import generate_tropical_cyclone_analysis

def generate_climate_hazards_analysis(shapefile_path, dbf_path, shx_path, water_risk_csv_path,
                                      facility_csv_path, raster_path, selected_fields=None,
                                      water_dynamic_fields=None, water_plot_fields=None):
    """
    Integrates water stress, flood exposure, sea level rise, and tropical cyclone analyses.
    
    Final output CSV will have these fields in order:
      Facility, Lat, Long, Exposure, bws_06_lab, SRTM elevation, 
      2030 Sea Level Rise Cl 0.5, 2040 Sea Level Rise Cl 0.5, 2050 Sea Level Rise Cl 0.5, 2060 Sea Level Rise Cl 0.5,
      1-min MSW 10 yr RP, 1-min MSW 20 yr RP, 1-min MSW 50 yr RP, 1-min MSW 100 yr RP
    """
    try:
        # ---------- WATER STRESS ANALYSIS ----------
        print("Step 1: Validating input files for water stress analysis")
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
        if 'PFAF_ID' not in hydrobasins.columns:
            raise ValueError("PFAF_ID not found in shapefile")
        if 'PFAF_ID' not in water_risk_data.columns:
            raise ValueError("PFAF_ID not found in water risk CSV")
        
        print("Step 4: Merging water risk data with shapefile")
        merged_data = hydrobasins.merge(water_risk_data, on='PFAF_ID', suffixes=('_hydro', '_risk'))
        ws_uploads_dir = os.path.join(settings.BASE_DIR, 'water_stress', 'static', 'input_files')
        os.makedirs(ws_uploads_dir, exist_ok=True)
        merged_shp_path = os.path.join(ws_uploads_dir, 'merged_aqueduct.shp')
        merged_data.to_file(merged_shp_path)
        print(f"Merged shapefile saved at {merged_shp_path}")
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
        geometry = [Point(xy) for xy in zip(facility_locs['Long'], facility_locs['Lat'])]
        facility_gdf = gpd.GeoDataFrame(facility_locs, geometry=geometry, crs="EPSG:4326")
        buffer_size = 0.0009  # ~100m square buffer
        facility_gdf['geometry'] = facility_gdf.geometry.apply(
            lambda pt: box(pt.x - buffer_size, pt.y - buffer_size, pt.x + buffer_size, pt.y + buffer_size)
        )
        # Restrict to a region if needed
        min_lat, max_lat = 0, 20
        min_lon, max_lon = 114, 130
        gdf = gdf.cx[min_lon:max_lon, min_lat:max_lat].reset_index(drop=True)
        
        print("Step 6: Spatial join to add water stress data")
        if water_dynamic_fields is None:
            water_dynamic_fields = ['bws_06_lab']
        for field in water_dynamic_fields:
            if field not in gdf.columns:
                raise ValueError(f"Field {field} not found in water risk data.")
        facility_gdf = gpd.sjoin(facility_gdf, gdf[['geometry'] + water_dynamic_fields],
                                  how='left', predicate='intersects').reset_index(drop=True)
        for field in water_dynamic_fields:
            facility_locs[field] = facility_gdf[field]
        updated_facility_csv = os.path.join(ws_uploads_dir, 'combined_facility_ws.csv')
        facility_locs.to_csv(updated_facility_csv, index=False)
        print(f"Updated facility CSV with water stress data saved at {updated_facility_csv}")
        
        # ---------- WATER STRESS PLOT ----------
        print("Step 7: Generating water stress plot")
        fig, ax = plt.subplots(figsize=(12, 8))
        gdf.boundary.plot(ax=ax, linewidth=1, color='black')
        for field in (water_plot_fields or water_dynamic_fields):
            if field in gdf.columns:
                gdf.plot(column=field, ax=ax, legend=True, cmap='OrRd')
            else:
                print(f"Warning: {field} not found for plotting")
        facility_gdf.plot(ax=ax, color='blue', marker='o', markersize=50, alpha=0.5)
        ax.set_title('Water Stress and Facility Locations', fontsize=15)
        ax.set_axis_off()
        plot_path = os.path.join(ws_uploads_dir, 'water_stress_plot.png')
        plt.savefig(plot_path, format='png')
        plt.close(fig)
        print(f"Water stress plot saved at {plot_path}")
        
        # ---------- FLOOD EXPOSURE ANALYSIS ----------
        print("Step 8: Performing flood exposure analysis")
        df_fac = pd.read_csv(updated_facility_csv)
        # Ensure we have a facility identifier; if missing, rename the first column to 'Site'
        for col in ['Long', 'Lat', 'Site']:
            if col not in df_fac.columns:
                df_fac.rename(columns={df_fac.columns[0]: 'Site'}, inplace=True)
        flood_buffer = 0.0045  # ~250m radius
        crs_flood = CRS("epsg:32651")
        geometry = [Point(xy).buffer(flood_buffer, cap_style=3) for xy in zip(df_fac['Long'], df_fac['Lat'])]
        flood_gdf = gpd.GeoDataFrame(df_fac, crs=crs_flood, geometry=geometry)
        stats = zonal_stats(flood_gdf.geometry, raster_path, stats="percentile_75")
        flood_gdf['75th Percentile'] = [stat.get('percentile_75', 1) if stat.get('percentile_75') is not None else 1 for stat in stats]
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
        
        # ---------- GENERATE COMBINED OUTPUT (Water Stress + Flood Exposure) ----------
        print("Step 9: Generating combined output table for water stress and flood exposure")
        selected_fields_mapping = {
            'Flood': 'Exposure',
            'Water Stress': 'bws_06_lab',
        }
        selected_fields = [selected_fields_mapping.get(field, field) for field in (selected_fields or list(selected_fields_mapping.keys()))]
        required_columns = ['Site', 'Lat', 'Long'] + selected_fields
        for col in required_columns:
            if col not in flood_gdf.columns:
                raise ValueError(f"Expected column '{col}' not found in combined data.")
        combined_df = flood_gdf[required_columns]
        combined_uploads_dir = os.path.join(settings.BASE_DIR, 'climate_hazards_analysis', 'static', 'input_files')
        os.makedirs(combined_uploads_dir, exist_ok=True)
        combined_csv_path = os.path.join(combined_uploads_dir, 'combined_output.csv')
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Combined output CSV (water stress and flood exposure) saved at {combined_csv_path}")
        
        # ---------- INTEGRATE SEA LEVEL RISE ANALYSIS ----------
        print("Step 10: Integrating sea level rise analysis")
        slr_result = generate_sea_level_rise_analysis(facility_csv_path)
        if "error" in slr_result:
            raise Exception("Error in sea level rise analysis: " + slr_result["error"])
        slr_csv_path = slr_result["combined_csv_paths"][0]
        df_slr = pd.read_csv(slr_csv_path)
        # Expecting columns such as "Facility", "Lat", "Lon", "SRTM elevation", "2030 Sea Level Rise CI 0.5", etc.
        median_years = [2030, 2040, 2050, 2060]
        median_columns = {}
        for year in median_years:
            orig_col = f"{year} Sea Level Rise CI 0.5"
            new_col = f"{year} Sea Level Rise Cl 0.5"
            if orig_col in df_slr.columns:
                median_columns[orig_col] = new_col
            else:
                print(f"Warning: Expected column {orig_col} not found in SLR results.")
        if "Lon" in df_slr.columns:
            df_slr.rename(columns={"Lon": "Long"}, inplace=True)
        slr_subset = df_slr[["Facility", "Lat", "Long", "SRTM elevation"] + list(median_columns.keys())].copy()
        slr_subset.rename(columns=median_columns, inplace=True)
        # For merging, rename "Facility" to "Site"
        slr_subset.rename(columns={"Facility": "Site"}, inplace=True)
        final_df = combined_df.merge(slr_subset, on=["Site", "Lat", "Long"], how="left")
        # Rename "Site" to "Facility" for the final output.
        final_df.rename(columns={"Site": "Facility"}, inplace=True)
        
        # ---------- INTEGRATE TROPICAL CYCLONE ANALYSIS ----------
        print("Step 11: Integrating tropical cyclone analysis")
        tc_result = generate_tropical_cyclone_analysis(facility_csv_path)
        if "error" in tc_result:
            raise Exception("Error in tropical cyclone analysis: " + tc_result["error"])
        tc_csv_path = tc_result["combined_csv_paths"][0]
        df_tc = pd.read_csv(tc_csv_path)
        # Rename columns for consistency.
        df_tc.rename(columns={"Facility Name": "Facility", "Latitude": "Lat", "Longitude": "Long"}, inplace=True)
        # Merge tropical cyclone results with final_df.
        final_df = final_df.merge(df_tc, on=["Facility", "Lat", "Long"], how="left")
        
        # ---------- FINAL OUTPUT ----------
        # Subset the final DataFrame so that it has only the required fields in the correct order.
        final_columns = [
            "Facility", 
            "Lat", 
            "Long", 
            "Exposure", 
            "bws_06_lab", 
            "SRTM elevation", 
            "2030 Sea Level Rise Cl 0.5", 
            "2040 Sea Level Rise Cl 0.5", 
            "2050 Sea Level Rise Cl 0.5", 
            "2060 Sea Level Rise Cl 0.5", 
            "1-min MSW 10 yr RP", 
            "1-min MSW 20 yr RP", 
            "1-min MSW 50 yr RP", 
            "1-min MSW 100 yr RP"
        ]
        missing = [col for col in final_columns if col not in final_df.columns]
        if missing:
            raise ValueError(f"Final output is missing required columns: {missing}")
        final_df = final_df[final_columns]
        
        final_csv_path = os.path.join(combined_uploads_dir, 'final_combined_output.csv')
        final_df.to_csv(final_csv_path, index=False)
        print(f"Final combined CSV saved at {final_csv_path}")
        
        return {
            'plot_path': plot_path,
            'combined_csv_path': final_csv_path
        }
        
    except Exception as e:
        print(f"Error in generate_climate_hazards_analysis: {e}")
        return None