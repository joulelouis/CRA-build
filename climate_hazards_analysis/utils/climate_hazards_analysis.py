import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from django.conf import settings
import rasterstats as rstat
import math

# Import all external hazard analyses
from sea_level_rise_analysis.utils.sea_level_rise_analysis import generate_sea_level_rise_analysis
from tropical_cyclone_analysis.utils.tropical_cyclone_analysis import generate_tropical_cyclone_analysis
from water_stress.utils.water_stress_analysis import generate_water_stress_analysis
from heat_exposure_analysis.utils.heat_exposure_analysis import generate_heat_exposure_analysis
from flood_exposure_analysis.utils.flood_exposure_analysis import generate_flood_exposure_analysis


def generate_climate_hazards_analysis(shapefile_path=None, dbf_path=None, shx_path=None,
                                     water_risk_csv_path=None, facility_csv_path=None,
                                     raster_path=None, selected_fields=None,
                                     water_dynamic_fields=None, water_plot_fields=None):
    """
    Integrates multiple climate hazard analyses into a single output.
    
    Args:
        shapefile_path: Path to water stress shapefile (optional, for backward compatibility)
        dbf_path: Path to water stress DBF file (optional, for backward compatibility)
        shx_path: Path to water stress SHX file (optional, for backward compatibility)
        water_risk_csv_path: Path to water risk CSV (optional, for backward compatibility)
        facility_csv_path: Path to facility locations CSV (required)
        raster_path: Path to flood raster (optional, for backward compatibility)
        selected_fields: List of selected hazard types to include
        water_dynamic_fields: Water stress dynamic fields (optional, for backward compatibility)
        water_plot_fields: Water stress plot fields (optional, for backward compatibility)
        
    Returns:
        dict: Results dictionary with paths to combined output and plots
    """
    try:
        if not facility_csv_path or not os.path.exists(facility_csv_path):
            raise ValueError("Facility CSV path is required and must exist.")
            
        if not selected_fields:
            selected_fields = []
            
        # Define path for final combined output
        input_dir = os.path.join(settings.BASE_DIR, 'climate_hazards_analysis', 'static', 'input_files')
        os.makedirs(input_dir, exist_ok=True)
        
        # Ensure the base facility CSV has consistent column names
        df_fac = pd.read_csv(facility_csv_path)
        
        # Standardize facility name column
        rename_map = {}
        for col in df_fac.columns:
            low = col.strip().lower()
            if low in ['facility', 'site', 'site name', 'facility name', 'facilty name']:
                rename_map[col] = 'Facility'
        if rename_map:
            df_fac.rename(columns=rename_map, inplace=True)
            
        # Standardize lat/long columns
        for old, new in [('latitude', 'Lat'), ('longitude', 'Long')]:
            if old in df_fac.columns and new not in df_fac.columns:
                df_fac.rename(columns={old: new}, inplace=True)
        
        # Ensure Facility and coordinates are present
        required_cols = ['Facility', 'Lat', 'Long']
        missing = [col for col in required_cols if col not in df_fac.columns]
        if missing:
            raise ValueError(f"Missing required columns in facility CSV: {', '.join(missing)}")
        
        # Convert coordinates to numeric and drop invalid
        df_fac['Lat'] = pd.to_numeric(df_fac['Lat'], errors='coerce')
        df_fac['Long'] = pd.to_numeric(df_fac['Long'], errors='coerce')
        df_fac.dropna(subset=['Lat', 'Long'], inplace=True)
        
        if df_fac.empty:
            raise ValueError("No valid facility locations after processing.")
        
        # Initialize combined DataFrame with base columns
        combined_df = df_fac[['Facility', 'Lat', 'Long']].copy()
        
        # Track plots for visualization
        plot_paths = []
        
        # Create variables to store the actual values for later merging
        flood_exposure_values = None
        water_stress_values = None
        slr_values = None
        tc_values = None
        heat_values = None
        ss_ril_values = None

        # ---------- 1. FLOOD EXPOSURE ANALYSIS ----------
        if 'Flood' in selected_fields:
            print("Integrating Flood Exposure Analysis")
            flood_res = generate_flood_exposure_analysis(facility_csv_path)
            if 'error' in flood_res:
                print(f"Warning in Flood Exposure Analysis: {flood_res['error']}")
            else:
                if flood_res.get('combined_csv_paths'):
                    df_flood = pd.read_csv(flood_res['combined_csv_paths'][0])
                    # Ensure column name consistency
                    for old, new in [('Site', 'Facility'), ('latitude', 'Lat'), ('longitude', 'Long')]:
                        if old in df_flood.columns and new not in df_flood.columns:
                            df_flood.rename(columns={old: new}, inplace=True)
                    
                    # Get flood exposure values
                    if 'Flood Depth (meters)' in df_flood.columns:
                        # Get flood values
                        df_flood_values = df_flood[['Facility', 'Lat', 'Long', 'Flood Depth (meters)']]
                        flood_exposure_values = df_flood_values.copy()
                    elif 'Exposure' in df_flood.columns:
                        # Fall back to old column name if needed
                        df_flood.rename(columns={'Exposure': 'Flood Depth (meters)'}, inplace=True)
                        df_flood_values = df_flood[['Facility', 'Lat', 'Long', 'Flood Depth (meters)']]
                        flood_exposure_values = df_flood_values.copy()
                        
                if flood_res.get('png_paths'):
                    plot_paths.extend(flood_res['png_paths'])

        # ---------- 2. WATER STRESS ANALYSIS ----------
        if 'Water Stress' in selected_fields:
            print("Integrating Water Stress Analysis")
            ws_res = generate_water_stress_analysis(facility_csv_path)
            if 'error' in ws_res:
                print(f"Warning in Water Stress Analysis: {ws_res['error']}")
            else:
                if ws_res.get('combined_csv_paths'):
                    df_ws = pd.read_csv(ws_res['combined_csv_paths'][0])
                    # Ensure column name consistency
                    for old, new in [('Site', 'Facility'), ('latitude', 'Lat'), ('longitude', 'Long')]:
                        if old in df_ws.columns and new not in df_ws.columns:
                            df_ws.rename(columns={old: new}, inplace=True)
                    
                    # Get water stress values
                    if 'Water Stress Exposure (%)' in df_ws.columns:
                        # Merge into the main DataFrame with a temporary name
                        df_ws_values = df_ws[['Facility', 'Lat', 'Long', 'Water Stress Exposure (%)']]
                        water_stress_values = df_ws_values.copy()
                    elif 'bws_06_raw' in df_ws.columns:
                        # Fall back to old column name if needed
                        df_ws.rename(columns={'bws_06_raw': 'Water Stress Exposure (%)'}, inplace=True)
                        df_ws_values = df_ws[['Facility', 'Lat', 'Long', 'Water Stress Exposure (%)']]
                        water_stress_values = df_ws_values.copy()
                        
                if ws_res.get('png_paths'):
                    plot_paths.extend(ws_res['png_paths'])

        # ---------- 3. SEA LEVEL RISE ANALYSIS ----------
        if 'Sea Level Rise' in selected_fields:
            print("Integrating Sea Level Rise Analysis")
            slr_res = generate_sea_level_rise_analysis(facility_csv_path)
            if 'error' in slr_res:
                print(f"Warning in Sea Level Rise Analysis: {slr_res['error']}")
            else:
                if slr_res.get('combined_csv_paths'):
                    df_slr = pd.read_csv(slr_res['combined_csv_paths'][0])
                    # Ensure column name consistency
                    for old, new in [('Site', 'Facility'), ('Lon', 'Long')]:
                        if old in df_slr.columns and new not in df_slr.columns:
                            df_slr.rename(columns={old: new}, inplace=True)
                    # Standardize Sea Level Rise column names
                    ren = {
                        f"{yr} Sea Level Rise CI 0.5": f"{yr} Sea Level Rise (in meters)"
                        for yr in [2030, 2040, 2050, 2060]
                    }
                    df_slr.rename(columns=ren, inplace=True)
                    # Also rename SRTM elevation
                    if 'SRTM elevation' in df_slr.columns:
                        df_slr.rename(columns={'SRTM elevation': 'Elevation (meter above sea level)'}, inplace=True)
                    
                    # Get SLR columns
                    slr_cols = ['Elevation (meter above sea level)'] + list(ren.values())
                    available_slr_cols = [c for c in slr_cols if c in df_slr.columns]
                    
                    if available_slr_cols:
                        # Store SLR values for later
                        slr_values = df_slr[['Facility', 'Lat', 'Long'] + available_slr_cols].copy()
                if slr_res.get('png_paths'):
                    plot_paths.extend(slr_res['png_paths'])

        # ---------- 4. TROPICAL CYCLONES ANALYSIS ----------
        if 'Tropical Cyclones' in selected_fields:
            print("Integrating Tropical Cyclones Analysis")
            tc_res = generate_tropical_cyclone_analysis(facility_csv_path)
            if 'error' in tc_res:
                print(f"Warning in Tropical Cyclones Analysis: {tc_res['error']}")
            else:
                if tc_res.get('combined_csv_paths'):
                    df_tc = pd.read_csv(tc_res['combined_csv_paths'][0])
                    # Ensure column name consistency
                    rename_map = {
                        'Facility Name': 'Facility',
                        'Latitude': 'Lat',
                        'Longitude': 'Long'
                    }
                    for old, new in rename_map.items():
                        if old in df_tc.columns and new not in df_tc.columns:
                            df_tc.rename(columns={old: new}, inplace=True)
                    
                    # Standardize TC column names
                    tc_rename = {
                        '1-min MSW 10 yr RP': '1-min Maximum Sustain Windspeed 10 year Return Period (km/h)',
                        '1-min MSW 20 yr RP': '1-min Maximum Sustain Windspeed 20 year Return Period (km/h)',
                        '1-min MSW 50 yr RP': '1-min Maximum Sustain Windspeed 50 year Return Period (km/h)',
                        '1-min MSW 100 yr RP': '1-min Maximum Sustain Windspeed 100 year Return Period (km/h)'
                    }
                    df_tc.rename(columns=tc_rename, inplace=True)
                    
                    # Identify TC columns (exclude 200yr and 500yr if present)
                    tc_cols = [col for col in df_tc.columns if 'Maximum Sustain Windspeed' in col]
                    
                    if tc_cols:
                        # Store TC values for later
                        tc_values = df_tc[['Facility', 'Lat', 'Long'] + tc_cols].copy()

        # ---------- 5. HEAT EXPOSURE ANALYSIS ----------
        if 'Heat' in selected_fields:
            print("Integrating Heat Exposure Analysis")
            heat_res = generate_heat_exposure_analysis(facility_csv_path)
            if 'error' in heat_res:
                print(f"Warning in Heat Exposure Analysis: {heat_res['error']}")
            else:
                if heat_res.get('combined_csv_paths'):
                    df_heat = pd.read_csv(heat_res['combined_csv_paths'][0])
                    print(f"Heat exposure CSV loaded: {heat_res['combined_csv_paths'][0]}")
                    print(f"Heat exposure columns: {df_heat.columns.tolist()}")
                    
                    # Ensure column name consistency
                    for old, new in [('Site', 'Facility'), ('latitude', 'Lat'), ('longitude', 'Long')]:
                        if old in df_heat.columns and new not in df_heat.columns:
                            df_heat.rename(columns={old: new}, inplace=True)
                    
                    # Standardize heat column names from either format
                    heat_cols = []
                    # Handle original format from heat_exposure_analysis.py
                    original_heat_cols = [c for c in df_heat.columns if c.startswith('n>') and c.endswith('degC_2125')]
                    if original_heat_cols:
                        temp_mapping = {
                            'n>30degC_2125': 'Days over 30° Celsius',
                            'n>33degC_2125': 'Days over 33° Celsius',
                            'n>35degC_2125': 'Days over 35° Celsius'
                        }
                        for col in original_heat_cols:
                            if col in temp_mapping:
                                heat_cols.append(temp_mapping[col])
                                df_heat.rename(columns={col: temp_mapping[col]}, inplace=True)
                            else:
                                heat_cols.append(col)
                    
                    # Also check for already renamed columns
                    for col_name in ['Days over 30° Celsius', 'Days over 33° Celsius', 'Days over 35° Celsius']:
                        if col_name in df_heat.columns and col_name not in heat_cols:
                            heat_cols.append(col_name)
                    
                    print(f"Heat columns after processing: {heat_cols}")
                    
                    if heat_cols:
                        # Store heat data for later merging
                        heat_values = df_heat[['Facility', 'Lat', 'Long'] + heat_cols].copy()
                if heat_res.get('png_paths'):
                    plot_paths.extend(heat_res['png_paths'])

        # ---------- 6-7. STORM SURGE & RAINFALL INDUCED LANDSLIDE ----------
        # These are still embedded in the main function since they haven't been modularized yet
        if any(h in selected_fields for h in ['Storm Surge', 'Rainfall Induced Landslide']):
            print("Integrating Storm Surge & Rainfall Induced Landslide Analyses")
            try:
                # Paths to raster files
                idir = Path(settings.BASE_DIR)/'climate_hazards_analysis'/'static'/'input_files'
                fp_ls = idir/'PH_LandslideHazards_UTM_ProjectNOAH_Unmasked.tif'
                fp_ss = idir/'PH_StormSurge_Advisory4_UTM_ProjectNOAH_Unmasked.tif'
                
                # Check if raster files exist
                if ('Storm Surge' in selected_fields and not os.path.exists(fp_ss)) or \
                   ('Rainfall Induced Landslide' in selected_fields and not os.path.exists(fp_ls)):
                    missing_files = []
                    if 'Storm Surge' in selected_fields and not os.path.exists(fp_ss):
                        missing_files.append(str(fp_ss))
                    if 'Rainfall Induced Landslide' in selected_fields and not os.path.exists(fp_ls):
                        missing_files.append(str(fp_ls))
                    print(f"Warning: Missing raster files for SS/RIL analysis: {', '.join(missing_files)}")
                else:
                    # Create GeoDataFrame with points
                    df_a = df_fac[['Facility', 'Lat', 'Long']].copy()
                    df_a.rename(columns={'Lat': 'latitude', 'Long': 'longitude'}, inplace=True)
                    
                    # Convert to numeric
                    df_a[['latitude', 'longitude']] = df_a[['latitude', 'longitude']].astype(float)
                    
                    # Set buffer parameters
                    df_a['lot_area'] = 250
                    
                    # Create geometry from points
                    gs_a = gpd.points_from_xy(df_a['longitude'], df_a['latitude'], crs='EPSG:4326').to_crs('EPSG:32651')
                    gdf_a = gpd.GeoDataFrame(df_a, geometry=gs_a, crs='EPSG:32651')
                    gdf_a['geometry'] = gdf_a.geometry.buffer(np.sqrt(gdf_a['lot_area'])/2, cap_style='square', join_style='mitre')
                    
                    # Process each raster file if selected and present
                    for lbl, ras, hazard_type in [
                        ('stormsurge_raster', fp_ss, 'Storm Surge'),  # Storm Surge first
                        ('landslide_raster', fp_ls, 'Rainfall Induced Landslide')  # Rainfall Induced Landslide second
                    ]:
                        if hazard_type in selected_fields and os.path.exists(str(ras)):
                            stats = rstat.zonal_stats(gdf_a, ras, stats='percentile_75', nodata=255)
                            gdf_a[lbl] = pd.DataFrame(stats)['percentile_75'].fillna(0)
                        elif hazard_type in selected_fields:
                            print(f"Warning: {hazard_type} raster not found: {ras}")
                            gdf_a[lbl] = 0  # Default value
                    
                    # Rename columns back for merging
                    gdf_a.rename(columns={'latitude': 'Lat', 'longitude': 'Long'}, inplace=True)
                    
                    # Store SS/RIL values
                    hazard_cols = []
                    if 'Storm Surge' in selected_fields and 'stormsurge_raster' in gdf_a.columns:
                        gdf_a.rename(columns={'stormsurge_raster': 'Storm Surge Hazard Rating'}, inplace=True)
                        hazard_cols.append('Storm Surge Hazard Rating')
                    
                    if 'Rainfall Induced Landslide' in selected_fields and 'landslide_raster' in gdf_a.columns:
                        gdf_a.rename(columns={'landslide_raster': 'Rainfall Induced Landslide Hazard Rating'}, inplace=True)
                        hazard_cols.append('Rainfall Induced Landslide Hazard Rating')
                        
                    if hazard_cols:
                        ss_ril_values = gdf_a[['Facility', 'Lat', 'Long'] + hazard_cols].copy()
            except Exception as e:
                print(f"Warning in Storm Surge/Landslide analysis: {e}")

        # ---------- Now add all hazard data to combined DataFrame in the correct order ----------
        
        # 1. Add Flood Exposure data
        if flood_exposure_values is not None:
            print(f"Merging flood exposure values with {len(flood_exposure_values)} rows")
            combined_df = combined_df.merge(flood_exposure_values, on=['Facility', 'Lat', 'Long'], how='left')
            
        # 2. Add Water Stress data
        if water_stress_values is not None:
            print(f"Merging water stress values with {len(water_stress_values)} rows")
            combined_df = combined_df.merge(water_stress_values, on=['Facility', 'Lat', 'Long'], how='left')
            
        # 3. Add Sea Level Rise data
        if slr_values is not None:
            print(f"Merging SLR values with {len(slr_values)} rows")
            combined_df = combined_df.merge(slr_values, on=['Facility', 'Lat', 'Long'], how='left')
            
        # 4. Add Tropical Cyclones data
        if tc_values is not None:
            print(f"Merging TC values with {len(tc_values)} rows")
            combined_df = combined_df.merge(tc_values, on=['Facility', 'Lat', 'Long'], how='left')
            
        # 5. Add Heat Exposure data
        if heat_values is not None:
            print(f"Merging heat values with {len(heat_values)} rows")
            combined_df = combined_df.merge(heat_values, on=['Facility', 'Lat', 'Long'], how='left')
            
        # 6-7. Add Storm Surge / Rainfall Induced Landslide data
        if ss_ril_values is not None:
            print(f"Merging SS/RIL values with {len(ss_ril_values)} rows")
            combined_df = combined_df.merge(ss_ril_values, on=['Facility', 'Lat', 'Long'], how='left')

        # ---------- PROCESS DATA FOR NAN VALUES ----------
        # Replace NaN values with custom strings
        for col in combined_df.columns:
            if col in ['Facility', 'Lat', 'Long']:
                continue
                
            if 'Sea Level Rise' in col or col == 'Elevation (meter above sea level)':
                combined_df[col] = combined_df[col].apply(
                    lambda v: "Little to no effect" if pd.isna(v) else v
                )
            elif 'Maximum Sustain Windspeed' in col:
                combined_df[col] = combined_df[col].apply(
                    lambda v: "Data not available" if pd.isna(v) else v
                )
            else:
                combined_df[col] = combined_df[col].apply(
                    lambda v: "N/A" if pd.isna(v) else v
                )

        # ---------- FINAL OUTPUT ----------
        print("Writing combined output CSV")
        
        # Save the combined output to CSV
        out_csv = os.path.join(input_dir, 'combined_output.csv')
        combined_df.to_csv(out_csv, index=False)
        print(f"Saved combined output CSV: {out_csv}")
        
        # Select the main plot for display (prioritizing flood exposure if available)
        main_plot = None
        if plot_paths:
            # Prioritize flood exposure plot if available
            flood_plots = [p for p in plot_paths if 'flood_exposure' in p.lower()]
            if flood_plots:
                main_plot = flood_plots[0]
            else:
                main_plot = plot_paths[0]
        
        return {
            'combined_csv_path': out_csv,
            'plot_path': main_plot,
            'all_plots': plot_paths
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print full stack trace for debugging
        return {'error': str(e), 'combined_csv_path': None, 'plot_path': None}  # Return empty paths