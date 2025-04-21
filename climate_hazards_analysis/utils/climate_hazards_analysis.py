# import os
# from pathlib import Path
# import geopandas as gpd
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from shapely.geometry import Point, box
# from django.conf import settings
# from rasterstats import zonal_stats
# from pyproj import CRS
# import time
# import re
# import rasterstats as rstat
# import math

# # Import external hazard analyses.
# from sea_level_rise_analysis.utils.sea_level_rise_analysis import generate_sea_level_rise_analysis
# from tropical_cyclone_analysis.utils.tropical_cyclone_analysis import generate_tropical_cyclone_analysis

# def generate_climate_hazards_analysis(shapefile_path, dbf_path, shx_path, water_risk_csv_path,
#                                       facility_csv_path, raster_path, selected_fields=None,
#                                       water_dynamic_fields=None, water_plot_fields=None):
#     """
#     Integrates water stress, flood exposure, sea level rise, tropical cyclone,
#     and (optionally) heat exposure analyses.
    
#     When 'Heat' is among the selected_fields, the output CSV will also include:
#       n>30degC_2125, n>33degC_2125, n>35degC_2125.
#     The heat exposure values are rounded upward and converted to whole numbers.
    
#     Final output CSV will have a base set of columns (Facility, Lat, Long, Exposure, etc.)
#     with additional hazard columns merged in.
#     """
#     try:
#         # ---------- WATER STRESS ANALYSIS ----------
#         print("Step 1: Validating input files for water stress analysis")
#         required_files = [shapefile_path, dbf_path, shx_path]
#         missing_files = [f for f in required_files if not os.path.exists(f)]
#         if missing_files:
#             raise FileNotFoundError(f"Missing shapefile components: {', '.join(missing_files)}")
#         if not os.path.exists(water_risk_csv_path):
#             raise FileNotFoundError(f"Water risk CSV {water_risk_csv_path} not found!")
#         if not os.path.exists(facility_csv_path):
#             raise FileNotFoundError(f"Facility CSV {facility_csv_path} not found!")
        
#         print("Step 2: Loading shapefile")
#         hydrobasins = gpd.read_file(shapefile_path)
#         if hydrobasins.crs is None:
#             print("WARNING: Shapefile has no CRS! Setting to EPSG:4326...")
#             hydrobasins.set_crs("EPSG:4326", inplace=True)
#         print(f"Shapefile CRS: {hydrobasins.crs}")
        
#         print("Step 3: Loading water risk CSV")
#         water_risk_data = pd.read_csv(water_risk_csv_path)
#         if 'PFAF_ID' not in hydrobasins.columns:
#             raise ValueError("PFAF_ID not found in shapefile")
#         if 'PFAF_ID' not in water_risk_data.columns:
#             raise ValueError("PFAF_ID not found in water risk CSV")
        
#         print("Step 4: Merging water risk data with shapefile")
#         merged_data = hydrobasins.merge(water_risk_data, on='PFAF_ID', suffixes=('_hydro', '_risk'))
#         ws_uploads_dir = os.path.join(settings.BASE_DIR, 'water_stress', 'static', 'input_files')
#         os.makedirs(ws_uploads_dir, exist_ok=True)
#         merged_shp_path = os.path.join(ws_uploads_dir, 'merged_aqueduct.shp')
#         merged_data.to_file(merged_shp_path)
#         print(f"Merged shapefile saved at {merged_shp_path}")
#         gdf = gpd.read_file(merged_shp_path)
        
#         # ---------- FACILITY DATA PREPARATION ----------
#         print("Step 5: Loading and cleaning facility CSV")
#         facility_locs = pd.read_csv(facility_csv_path)
#         for col in ['Long', 'Lat']:
#             if col not in facility_locs.columns:
#                 raise ValueError(f"Missing '{col}' column in facility CSV.")
#         facility_locs['Long'] = pd.to_numeric(facility_locs['Long'], errors='coerce')
#         facility_locs['Lat'] = pd.to_numeric(facility_locs['Lat'], errors='coerce')
#         facility_locs = facility_locs.dropna(subset=['Long', 'Lat'])
#         if facility_locs.empty:
#             raise ValueError("Facility CSV is empty after cleaning!")
#         # Ensure a consistent facility identifier.
#         if 'Facility' not in facility_locs.columns:
#             if 'Site' in facility_locs.columns:
#                 facility_locs.rename(columns={'Site': 'Facility'}, inplace=True)
#             elif 'Facilty Name' in facility_locs.columns:
#                 facility_locs.rename(columns={'Facilty Name': 'Facility'}, inplace=True)
#             else:
#                 facility_locs['Facility'] = facility_locs.apply(lambda row: f"Facility_{row.name}", axis=1)
        
#         geometry = [Point(xy) for xy in zip(facility_locs['Long'], facility_locs['Lat'])]
#         facility_gdf = gpd.GeoDataFrame(facility_locs, geometry=geometry, crs="EPSG:4326")
#         buffer_size = 0.0009  # ~100m square buffer
#         facility_gdf['geometry'] = facility_gdf.geometry.apply(
#             lambda pt: box(pt.x - buffer_size, pt.y - buffer_size, pt.x + buffer_size, pt.y + buffer_size)
#         )
#         # Restrict to a region if needed.
#         min_lat, max_lat = 0, 20
#         min_lon, max_lon = 114, 130
#         gdf = gdf.cx[min_lon:max_lon, min_lat:max_lat].reset_index(drop=True)
        
#         print("Step 6: Spatial join to add water stress data")

#         # Multiply bws_06_raw values by 100 and round up
#         gdf['bws_06_raw'] = gdf['bws_06_raw'].astype(float).apply(lambda x: math.ceil(x * 100))

#         if water_dynamic_fields is None:
#             water_dynamic_fields = ['bws_06_raw'] 

#         for field in water_dynamic_fields:
#             if field not in gdf.columns:
#                 raise ValueError(f"Field {field} not found in water risk data.")

#         facility_gdf = gpd.sjoin(
#             facility_gdf,
#             gdf[['geometry'] + water_dynamic_fields],
#             how='left',
#             predicate='intersects'
#         ).reset_index(drop=True)

#         for field in water_dynamic_fields:
#             facility_locs[field] = facility_gdf[field]

#         updated_facility_csv = os.path.join(ws_uploads_dir, 'combined_facility_ws.csv')
#         facility_locs.to_csv(updated_facility_csv, index=False)
#         print(f"Updated facility CSV with water stress data saved at {updated_facility_csv}")
        
#         # ---------- WATER STRESS PLOT ----------
#         print("Step 7: Generating water stress plot")
#         fig, ax = plt.subplots(figsize=(12, 8))
#         gdf.boundary.plot(ax=ax, linewidth=1, color='black')
#         for field in (water_plot_fields or water_dynamic_fields):
#             if field in gdf.columns:
#                 gdf.plot(column=field, ax=ax, legend=True, cmap='OrRd')
#             else:
#                 print(f"Warning: {field} not found for plotting")
#         facility_gdf.plot(ax=ax, color='blue', marker='o', markersize=50, alpha=0.5)
#         ax.set_title('Water Stress and Facility Locations', fontsize=15)
#         ax.set_axis_off()
#         plot_path = os.path.join(ws_uploads_dir, 'water_stress_plot.png')
#         plt.savefig(plot_path, format='png')
#         plt.close(fig)
#         print(f"Water stress plot saved at {plot_path}")
        
#         # ---------- FLOOD EXPOSURE ANALYSIS ----------
#         print("Step 8: Performing flood exposure analysis")
#         df_fac = pd.read_csv(updated_facility_csv)
#         if 'Facility' not in df_fac.columns:
#             df_fac['Facility'] = df_fac.apply(lambda row: f"Facility_{row.name}", axis=1)
#         flood_buffer = 0.0045  # ~250m radius
#         crs_flood = CRS("epsg:32651")
#         geometry = [Point(xy).buffer(flood_buffer, cap_style=3) for xy in zip(df_fac['Long'], df_fac['Lat'])]
#         flood_gdf = gpd.GeoDataFrame(df_fac, crs=crs_flood, geometry=geometry)
#         stats = zonal_stats(flood_gdf.geometry, raster_path, stats="percentile_75")
#         flood_gdf['75th Percentile'] = [stat.get('percentile_75', 1) if stat.get('percentile_75') is not None else 1 for stat in stats]
#         def determine_exposure(val):
#             if val == 1:
#                 return '0.1 to 0.5'
#             elif val == 2:
#                 return '0.5 to 1.5'
#             elif val == 3:
#                 return 'Greater than 1.5'
#             else:
#                 return 'Unknown'
#         flood_gdf['Exposure'] = flood_gdf['75th Percentile'].apply(determine_exposure)
        
#         # ---------- GENERATE BASE COMBINED OUTPUT (Water Stress + Flood Exposure) ----------
#         print("Step 9: Generating combined output table for water stress and flood exposure")
#         base_mapping = {
#             'Flood': ['Exposure'],
#             'Water Stress': ['bws_06_raw']
#         }
#         base_selected = [hazard for hazard in (selected_fields or []) if hazard in base_mapping]
#         required_columns = ['Facility', 'Lat', 'Long']
#         for hazard in base_selected:
#             required_columns.extend(base_mapping[hazard])
        
#         for col in required_columns:
#             if col not in flood_gdf.columns:
#                 raise ValueError(f"Expected column '{col}' not found in combined data.")
#         combined_df = flood_gdf[required_columns]
        
#         # ---------- INTEGRATE SEA LEVEL RISE ANALYSIS (if selected) ----------
#         if 'Sea Level Rise' in (selected_fields or []):
#             print("Step 10: Integrating sea level rise analysis")
#             slr_result = generate_sea_level_rise_analysis(facility_csv_path)
#             if "error" in slr_result:
#                 raise Exception("Error in sea level rise analysis: " + slr_result["error"])
#             slr_csv_path = slr_result["combined_csv_paths"][0]
#             df_slr = pd.read_csv(slr_csv_path)
#             median_years = [2030, 2040, 2050, 2060]
#             median_columns = {}
#             for year in median_years:
#                 orig_col = f"{year} Sea Level Rise CI 0.5"
#                 new_col = f"{year} Sea Level Rise Cl 0.5"
#                 if orig_col in df_slr.columns:
#                     median_columns[orig_col] = new_col
#                 else:
#                     print(f"Warning: Expected column {orig_col} not found in SLR results.")
#             if "Lon" in df_slr.columns:
#                 df_slr.rename(columns={"Lon": "Long"}, inplace=True)
#             slr_subset = df_slr[["Facility", "Lat", "Long", "SRTM elevation"] + list(median_columns.keys())].copy()
#             slr_subset.rename(columns=median_columns, inplace=True)
#             combined_df = combined_df.merge(slr_subset, on=["Facility", "Lat", "Long"], how="left")
        
#         # ---------- INTEGRATE TROPICAL CYCLONE ANALYSIS (if selected) ----------
#         if 'Tropical Cyclones' in (selected_fields or []):
#             print("Step 11: Integrating tropical cyclone analysis")
#             tc_result = generate_tropical_cyclone_analysis(facility_csv_path)
#             if "error" in tc_result:
#                 raise Exception("Error in tropical cyclone analysis: " + tc_result["error"])
#             tc_csv_path = tc_result["combined_csv_paths"][0]
#             df_tc = pd.read_csv(tc_csv_path)
#             df_tc.rename(columns={"Facility Name": "Facility", "Latitude": "Lat", "Longitude": "Long"}, inplace=True)
#             combined_df = combined_df.merge(df_tc, on=["Facility", "Lat", "Long"], how="left")
        
#         # ---------- INTEGRATE HEAT EXPOSURE ANALYSIS (if selected) ----------
#         if 'Heat' in (selected_fields or []):
#             input_files_dir = Path(settings.BASE_DIR) / "climate_hazards_analysis" / "static" / "input_files"
#             fps_daysover = [
#                 input_files_dir / "PH_DaysOver30degC_ANN_2021-2025.tif",
#                 input_files_dir / "PH_DaysOver33degC_ANN_2021-2025.tif",
#                 input_files_dir / "PH_DaysOver35degC_ANN_2021-2025.tif"
#             ]
#             print("Step 12: Integrating heat exposure analysis")
#             # Heat analysis uses the facility CSV as the base asset file.
#             df_heat = pd.read_csv(facility_csv_path)
#             df_heat = df_heat[~df_heat["Lat"].isna()]
#             if "Facility" not in df_heat.columns:
#                 if "Site" in df_heat.columns:
#                     df_heat.rename(columns={"Site": "Facility"}, inplace=True)
#                 else:
#                     df_heat["Facility"] = df_heat.apply(lambda row: f"Facility_{row.name}", axis=1)
#             gs_heat = gpd.points_from_xy(x=df_heat["Long"], y=df_heat["Lat"], crs="EPSG:4326").to_crs("EPSG:32651")
#             gdf_heat = gpd.GeoDataFrame(df_heat.copy(), geometry=gs_heat, crs="EPSG:32651")
#             if "lot_area" not in gdf_heat.columns:
#                 gdf_heat["lot_area"] = 1_000**2
#             gdf_heat["geometry"] = gdf_heat["geometry"].buffer(
#                 np.sqrt(gdf_heat["lot_area"]) / 2,
#                 cap_style="square",
#                 join_style="mitre"
#             )
#             heat_temp_cols = [f"n>{temp}degC_2125" for temp in [30, 33, 35]]
#             heat_temp_fps = fps_daysover
#             for col, fp in zip(heat_temp_cols, heat_temp_fps):
#                 temp_geojson = Path("temp_features.geojson")
#                 gdf_heat.to_crs("EPSG:4326").to_file(str(temp_geojson), driver="GeoJSON")
#                 out_geojson = rstat.zonal_stats(
#                     str(temp_geojson),
#                     str(fp),
#                     stats="percentile_75",
#                     all_touched=True,
#                     geojson_out=True
#                 )
#                 out_idxs = [int(feat["id"]) for feat in out_geojson]
#                 out_stat = [feat["properties"]["percentile_75"] for feat in out_geojson]
#                 gdf_heat[col] = pd.Series(data=out_stat, index=out_idxs)
#                 if temp_geojson.exists():
#                     temp_geojson.unlink()
#             heat_cols_to_keep = ["Facility", "Lat", "Long"] + heat_temp_cols
#             df_heat_final = gdf_heat[heat_cols_to_keep].copy()
#             # Merge heat exposure results with the combined output.
#             combined_df = combined_df.merge(df_heat_final, on=["Facility", "Lat", "Long"], how="left")
#             # Round up the heat exposure values and convert them to whole numbers.
#             for col in heat_temp_cols:
#                 combined_df[col] = combined_df[col].apply(lambda x: int(np.ceil(x)) if pd.notnull(x) else x)
        
#         # ---------- FINAL OUTPUT ----------
#         print("Step 13: Generating final combined output")
#         final_order = [
#             "Facility", "Lat", "Long",
#             "Exposure", "bws_06_raw",
#             "SRTM elevation", "2030 Sea Level Rise Cl 0.5", "2040 Sea Level Rise Cl 0.5",
#             "2050 Sea Level Rise Cl 0.5", "2060 Sea Level Rise Cl 0.5",
#             "1-min MSW 10 yr RP", "1-min MSW 20 yr RP", "1-min MSW 50 yr RP", "1-min MSW 100 yr RP"
#         ]
#         if 'Heat' in (selected_fields or []):
#             final_order.extend(heat_temp_cols)
        
#         final_columns = [col for col in final_order if col in combined_df.columns]
#         final_df = combined_df[final_columns]
        
#         combined_uploads_dir = os.path.join(settings.BASE_DIR, 'climate_hazards_analysis', 'static', 'input_files')
#         os.makedirs(combined_uploads_dir, exist_ok=True)
#         final_csv_path = os.path.join(combined_uploads_dir, 'final_combined_output.csv')
#         final_df.to_csv(final_csv_path, index=False)
#         print(f"Final combined CSV saved at {final_csv_path}")
        
#         return {
#             'plot_path': plot_path,
#             'combined_csv_path': final_csv_path
#         }
        
#     except Exception as e:
#         print(f"Error in generate_climate_hazards_analysis: {e}")
#         return {"error": str(e)}

import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
from django.conf import settings
from rasterstats import zonal_stats
from pyproj import CRS
import re
import rasterstats as rstat
import math

# Import external hazard analyses.
from sea_level_rise_analysis.utils.sea_level_rise_analysis import generate_sea_level_rise_analysis
from tropical_cyclone_analysis.utils.tropical_cyclone_analysis import generate_tropical_cyclone_analysis


def generate_climate_hazards_analysis(shapefile_path, dbf_path, shx_path,
                                      water_risk_csv_path, facility_csv_path,
                                      raster_path, selected_fields=None,
                                      water_dynamic_fields=None,
                                      water_plot_fields=None):
    """
    Integrates water stress, flood exposure, sea level rise, tropical cyclone,
    heat exposure, storm surge, and rainfall-induced landslide analyses.
    """
    try:
        # ---------- WATER STRESS ANALYSIS ----------
        print("Step 1: Validating input files for water stress analysis")
        # check shapefile parts
        required = [shapefile_path, dbf_path, shx_path]
        missing = [f for f in required if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(f"Missing shapefile components: {', '.join(missing)}")
        # check CSV inputs
        for f in (water_risk_csv_path, facility_csv_path):
            if not os.path.exists(f):
                raise FileNotFoundError(f"Required CSV not found: {f}")

        # Load shapefile
        print("Step 2: Loading shapefile")
        hydrobasins = gpd.read_file(shapefile_path)
        if hydrobasins.crs is None:
            hydrobasins.set_crs("EPSG:4326", inplace=True)

        # Load water risk data
        print("Step 3: Loading water risk CSV")
        water_risk = pd.read_csv(water_risk_csv_path)
        if 'PFAF_ID' not in hydrobasins.columns or 'PFAF_ID' not in water_risk.columns:
            raise ValueError("'PFAF_ID' must exist in both shapefile and water risk CSV")

        # Merge water risk
        print("Step 4: Merging water risk data")
        merged = hydrobasins.merge(water_risk, on='PFAF_ID', suffixes=('_hydro','_risk'))
        ws_dir = os.path.join(settings.BASE_DIR, 'water_stress', 'static', 'input_files')
        os.makedirs(ws_dir, exist_ok=True)
        shp_out = os.path.join(ws_dir, 'merged_aqueduct.shp')
        merged.to_file(shp_out)
        gdf = gpd.read_file(shp_out)

        # ---------- FACILITY DATA PREP ----------
        print("Step 5: Loading facility CSV and ensuring 'Facility' column")
        facility_locs = pd.read_csv(facility_csv_path)
        # robust header rename (case-insensitive)
        rename_map = {}
        for col in facility_locs.columns:
            low = col.strip().lower()
            if low in ['facility', 'site', 'site name', 'facility name', 'facilty name']:
                rename_map[col] = 'Facility'
        if rename_map:
            facility_locs.rename(columns=rename_map, inplace=True)

        # ensure coords
        for coord in ['Long','Lat']:
            if coord not in facility_locs.columns:
                raise ValueError(f"Missing '{coord}' column in facility CSV.")
        facility_locs['Long'] = pd.to_numeric(facility_locs['Long'], errors='coerce')
        facility_locs['Lat'] = pd.to_numeric(facility_locs['Lat'], errors='coerce')
        facility_locs.dropna(subset=['Long','Lat'], inplace=True)

        # require Facility column
        if 'Facility' not in facility_locs.columns:
            raise ValueError("Your facility CSV must include a 'Facility' column or equivalent header.")

        # buffer points (~100m)
        pts = [Point(x,y) for x,y in zip(facility_locs['Long'], facility_locs['Lat'])]
        facility_gdf = gpd.GeoDataFrame(facility_locs, geometry=pts, crs="EPSG:4326")
        buf = 0.0009
        facility_gdf['geometry'] = facility_gdf.geometry.apply(
            lambda p: box(p.x-buf, p.y-buf, p.x+buf, p.y+buf)
        )

        # clip water risk extent & normalize
        gdf = gdf.cx[114:130, 0:20].reset_index(drop=True)
        gdf['bws_06_raw'] = gdf['bws_06_raw'].astype(float).apply(lambda v: math.ceil(v*100))
        if water_dynamic_fields is None:
            water_dynamic_fields = ['bws_06_raw']
        for f in water_dynamic_fields:
            if f not in gdf.columns:
                raise ValueError(f"Field {f} missing in water risk data")

        # spatial join water stress
        print("Step 6: Spatial join water stress to facilities")
        facility_join = gpd.sjoin(
            facility_gdf, gdf[['geometry']+water_dynamic_fields],
            how='left', predicate='intersects'
        ).reset_index(drop=True)
        for f in water_dynamic_fields:
            facility_locs[f] = facility_join[f]
        updated_ws = os.path.join(ws_dir, 'combined_facility_ws.csv')
        facility_locs.to_csv(updated_ws, index=False)

        # plot water stress
        print("Step 7: Plotting water stress")
        fig, ax = plt.subplots(figsize=(12,8))
        gdf.boundary.plot(ax=ax, linewidth=1, color='black')
        for f in (water_plot_fields or water_dynamic_fields):
            if f in gdf:
                gdf.plot(column=f, ax=ax, legend=True, cmap='OrRd')
        facility_gdf.plot(ax=ax, color='blue', markersize=50, alpha=0.5)
        ax.set_axis_off(); ax.set_title('Water Stress and Facilities')
        plot_path = os.path.join(ws_dir, 'water_stress_plot.png')
        plt.savefig(plot_path, format='png'); plt.close(fig)

        # ---------- FLOOD EXPOSURE ANALYSIS ----------  
        print("Step 8: Flood exposure analysis")  
        df_fac = pd.read_csv(updated_ws)  
        if 'Facility' not in df_fac.columns:  
            raise ValueError("Combined facility CSV missing 'Facility'; check your upload headers.")  

        # build ~250 m buffers around each point  
        flood_buf = 0.0045  
        flood_polys = [Point(x, y).buffer(flood_buf, cap_style=3)  
                    for x, y in zip(df_fac['Long'], df_fac['Lat'])]  
        flood_gdf = gpd.GeoDataFrame(df_fac.copy(), geometry=flood_polys, crs=CRS('epsg:32651'))  

        # extract the 75th‐percentile flood depth  
        stats = zonal_stats(flood_gdf.geometry, raster_path, stats='percentile_75')  
        flood_gdf['75th Percentile'] = [stat['percentile_75'] if stat['percentile_75'] is not None else 1 for stat in stats]  

        # bin into your desired categories  
        def determine_exposure(percentile):  
            if pd.isna(percentile):  
                return 'Unknown'  
            if percentile == 1:  
                return '0.1 to 0.5'  
            elif percentile == 2:  
                return '0.5 to 1.5'  
            else:  
                return 'Greater than 1.5'  

        # apply and rename for clarity  
        flood_gdf['Exposure'] = flood_gdf['75th Percentile'].apply(determine_exposure) 

        # combine base
        print("Step 9: Combine water stress + flood exposure")
        mapping = {'Flood':['Exposure'], 'Water Stress':['bws_06_raw']}
        sel = [h for h in (selected_fields or []) if h in mapping]
        req_cols = ['Facility','Lat','Long'] + [c for h in sel for c in mapping[h]]
        combined_df = flood_gdf.loc[:, req_cols]

        # ---------- SEA LEVEL RISE ----------
        if 'Sea Level Rise' in (selected_fields or []):
            print("Integrate SLR")
            slr_res = generate_sea_level_rise_analysis(facility_csv_path)
            if 'error' in slr_res: raise Exception(slr_res['error'])
            df_slr = pd.read_csv(slr_res['combined_csv_paths'][0])
            if 'Lon' in df_slr: df_slr.rename(columns={'Lon':'Long'}, inplace=True)
            ren = {f"{yr} Sea Level Rise CI 0.5":f"{yr} Sea Level Rise Cl 0.5"
                   for yr in [2030,2040,2050,2060]}
            df_slr.rename(columns=ren, inplace=True)
            slr_cols = ['Facility','Lat','Long'] + list(ren.values())
            combined_df = combined_df.merge(df_slr[slr_cols], on=['Facility','Lat','Long'], how='left')

        # ---------- TROPICAL CYCLONES ----------
        if 'Tropical Cyclones' in (selected_fields or []):
            print("Integrate TC")
            tc_res = generate_tropical_cyclone_analysis(facility_csv_path)
            if 'error' in tc_res: raise Exception(tc_res['error'])
            df_tc = pd.read_csv(tc_res['combined_csv_paths'][0])
            df_tc.drop(['1-min MSW 200 yr RP', '1-min MSW 500 yr RP'],axis=1, errors='ignore', inplace=True)
            df_tc.rename(columns={'Facility Name':'Facility','Latitude':'Lat','Longitude':'Long'}, inplace=True)
            combined_df = combined_df.merge(df_tc, on=['Facility','Lat','Long'], how='left')

        # ---------- HEAT ----------
        if 'Heat' in (selected_fields or []):
            print("Integrate Heat")
            idir = Path(settings.BASE_DIR)/'climate_hazards_analysis'/'static'/'input_files'
            heat_files = [idir/f for f in [
                'PH_DaysOver30degC_ANN_2021-2025.tif',
                'PH_DaysOver33degC_ANN_2021-2025.tif',
                'PH_DaysOver35degC_ANN_2021-2025.tif']]
            
            # Use the same facility CSV as used earlier that already has 'Facility' column mapping
            df_heat = facility_locs[['Facility', 'Lat', 'Long']].copy()
            
            gs = gpd.points_from_xy(df_heat['Long'], df_heat['Lat'], crs='EPSG:4326').to_crs('EPSG:32651')
            gdf_heat = gpd.GeoDataFrame(df_heat, geometry=gs, crs='EPSG:32651')
            gdf_heat['lot_area'] = gdf_heat.get('lot_area', 1000**2)
            gdf_heat['geometry'] = gdf_heat.geometry.buffer(
                np.sqrt(gdf_heat['lot_area'])/2, cap_style='square', join_style='mitre')
            heat_cols = [f"n>{t}degC_2125" for t in [30,33,35]]
            temp_geo = Path('temp.features.geojson')
            for col, fp in zip(heat_cols, heat_files):
                gdf_heat.to_crs('EPSG:4326').to_file(str(temp_geo), driver='GeoJSON')
                out = rstat.zonal_stats(str(temp_geo), str(fp), stats='percentile_75', all_touched=True, geojson_out=True)
                if out:  # Check if we got results
                    idxs = [int(feat['id']) for feat in out]
                    vals = [feat['properties']['percentile_75'] for feat in out]
                    gdf_heat[col] = pd.Series(vals, index=idxs)
                else:
                    # Handle no results
                    for col in heat_cols:
                        gdf_heat[col] = np.nan
                
                # Clean up temp file if it exists
                if temp_geo.exists():
                    temp_geo.unlink()
                    
            df_hf = gdf_heat[['Facility','Lat','Long'] + [c for c in heat_cols if c in gdf_heat.columns]].copy()
            for col in [c for c in heat_cols if c in df_hf.columns]:
                df_hf.loc[:,col] = df_hf[col].apply(lambda v: int(math.ceil(v)) if pd.notnull(v) else v)
            combined_df = combined_df.merge(df_hf, on=['Facility','Lat','Long'], how='left')

        # ---------- STORM SURGE & RIL ----------
        if any(h in (selected_fields or []) for h in ['Storm Surge','Rainfall Induced Landslide']):
            print("Integrate RIL & SS")
            idir = Path(settings.BASE_DIR)/'climate_hazards_analysis'/'static'/'input_files'
            fp_ls = idir/'PH_LandslideHazards_UTM_ProjectNOAH_Unmasked.tif'
            fp_ss = idir/'PH_StormSurge_Advisory4_UTM_ProjectNOAH_Unmasked.tif'
            
            # Use facility_locs which already has the correct 'Facility' column
            df_a = facility_locs[['Facility','Lat','Long']].copy()
            df_a.rename(columns={'Lat':'latitude','Long':'longitude'}, inplace=True)
            
            df_a[['latitude','longitude']] = df_a[['latitude','longitude']].astype(float)
            df_a['lot_area'] = 250
            gs_a = gpd.points_from_xy(df_a['longitude'], df_a['latitude'], crs='EPSG:4326').to_crs('EPSG:32651')
            gdf_a = gpd.GeoDataFrame(df_a, geometry=gs_a, crs='EPSG:32651')
            gdf_a['geometry'] = gdf_a.geometry.buffer(np.sqrt(gdf_a['lot_area'])/2, cap_style='square', join_style='mitre')
            for lbl, ras in [('landslide_raster', fp_ls), ('stormsurge_raster', fp_ss)]:
                # Check if raster file exists
                if os.path.exists(str(ras)):
                    stats = rstat.zonal_stats(gdf_a, ras, stats='percentile_75', nodata=255)
                    gdf_a[lbl] = pd.DataFrame(stats)['percentile_75'].fillna(0)
                else:
                    print(f"Warning: Raster file not found: {ras}")
                    gdf_a[lbl] = 0  # Default value
                    
            gdf_a.rename(columns={'latitude':'Lat','longitude':'Long'}, inplace=True)
            if 'Storm Surge' in (selected_fields or []):
                combined_df = combined_df.merge(
                    gdf_a[['Facility','Lat','Long','stormsurge_raster']], on=['Facility','Lat','Long'], how='left'
                )
            if 'Rainfall Induced Landslide' in (selected_fields or []):
                combined_df = combined_df.merge(
                    gdf_a[['Facility','Lat','Long','landslide_raster']], on=['Facility','Lat','Long'], how='left'
                )

        # ---------- FINAL OUTPUT ----------
        print("Step 14: Writing output CSV")
        # Only include columns that actually exist in combined_df
        order = ['Facility','Lat','Long']
        
        if 'Exposure' in combined_df.columns:
            order.append('Exposure')
        if 'bws_06_raw' in combined_df.columns:
            order.append('bws_06_raw')
            
        slr_cols = [c for c in combined_df.columns if 'Sea Level Rise Cl' in c]
        if slr_cols:
            order.extend(slr_cols)
            
        tc_cols = [c for c in combined_df.columns if 'MSW' in c]
        if tc_cols:
            order.extend(tc_cols)
            
        if 'Heat' in (selected_fields or []):
            heat_cols = [c for c in combined_df.columns if 'degC_2125' in c]
            if heat_cols:
                order.extend(heat_cols)
                
        if 'Storm Surge' in (selected_fields or []) and 'stormsurge_raster' in combined_df.columns:
            order.append('stormsurge_raster')
            
        if 'Rainfall Induced Landslide' in (selected_fields or []) and 'landslide_raster' in combined_df.columns:
            order.append('landslide_raster')
            
        # Only select columns that exist in combined_df
        final_df = combined_df.loc[:, [c for c in order if c in combined_df.columns]]
        out_dir = os.path.join(settings.BASE_DIR, 'climate_hazards_analysis', 'static', 'input_files')
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, 'final_combined_output.csv')
        final_df.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")
        return {'plot_path': plot_path, 'combined_csv_path': out_csv}
    except Exception as e:
        print(f"Error in generate_climate_hazards_analysis: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace for debugging
        return {'error': str(e), 'combined_csv_path': None, 'plot_path': None}  # Return empty paths

