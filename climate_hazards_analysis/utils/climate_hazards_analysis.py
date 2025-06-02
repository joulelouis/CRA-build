"""
Integrated Climate Hazards Analysis Module

This module provides functionality to generate a combined analysis of multiple climate hazards
for facility locations, integrating data from specialized modules for each hazard type.

Dependencies:
- pandas, numpy, matplotlib, geopandas
- Specialized hazard analysis modules
"""

import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterstats as rstat
from django.conf import settings

# Import specialized hazard analysis modules
from sea_level_rise_analysis.utils.sea_level_rise_analysis import generate_sea_level_rise_analysis
from tropical_cyclone_analysis.utils.tropical_cyclone_analysis import generate_tropical_cyclone_analysis
from water_stress.utils.water_stress_analysis import generate_water_stress_analysis
from heat_exposure_analysis.utils.heat_exposure_analysis import generate_heat_exposure_analysis
from flood_exposure_analysis.utils.flood_exposure_analysis import generate_flood_exposure_analysis

# Set up logging
logger = logging.getLogger(__name__)


def standardize_facility_dataframe(df):
    """
    Standardize facility dataframe column names for consistency.
    
    Args:
        df (pandas.DataFrame): The input facility dataframe
        
    Returns:
        pandas.DataFrame: Standardized dataframe with consistent column names
    """
    df = df.copy()
    
    # Standardize facility name column
    facility_name_variations = [
        'facility', 'site', 'site name', 
        'facility name', 'facilty name', 'name',
        'asset name'  # Add this to recognize "Asset Name" column
    ]
    
    # Find and rename facility name column
    for col in df.columns:
        if col.strip().lower() in facility_name_variations:
            df.rename(columns={col: 'Facility'}, inplace=True)
            break
            
    # Standardize lat/long columns - make it case-insensitive
    coord_mapping = {'latitude': 'Lat', 'longitude': 'Long'}
    for old, new in coord_mapping.items():
        for col in df.columns:
            if col.lower() == old.lower() and new not in df.columns:
                df.rename(columns={col: new}, inplace=True)
    
    # Validate required columns
    required_cols = ['Facility', 'Lat', 'Long']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in facility CSV: {', '.join(missing)}")
    
    # Convert coordinates to numeric and drop invalid
    df['Lat'] = pd.to_numeric(df['Lat'], errors='coerce')
    df['Long'] = pd.to_numeric(df['Long'], errors='coerce')
    df.dropna(subset=['Lat', 'Long'], inplace=True)
    
    if df.empty:
        raise ValueError("No valid facility locations after processing.")
        
    return df


def process_flood_exposure_analysis(facility_csv_path, selected_fields, buffer_size=0.0045):
    """
    Process flood exposure analysis if selected.
    Args:
        facility_csv_path (str): Path to facility CSV
        selected_fields (list): List of selected hazard types
        buffer_size (float): Buffer size for analysis
        
    Returns:
        tuple: (DataFrame with flood values, list of plot paths)
    """
    if 'Flood' not in selected_fields:
        return None, []
        
    logger.info(f"Integrating Flood Exposure Analysis with buffer size: {buffer_size}")
    plot_paths = []

    try:
        flood_res = generate_flood_exposure_analysis(facility_csv_path, buffer_size)
        
        if 'error' in flood_res:
            logger.warning(f"Warning in Flood Exposure Analysis: {flood_res['error']}")
            return None, []
            
        if not flood_res.get('combined_csv_paths'):
            return None, plot_paths
            
        # Read the flood analysis CSV
        df_flood = pd.read_csv(flood_res['combined_csv_paths'][0])
        
        # Standardize column names
        rename_map = {'Site': 'Facility', 'latitude': 'Lat', 'longitude': 'Long'}
        for old, new in rename_map.items():
            if old in df_flood.columns and new not in df_flood.columns:
                df_flood.rename(columns={old: new}, inplace=True)
        
        # Handle flood depth column variations
        if 'Flood Depth (meters)' in df_flood.columns:
            df_flood_values = df_flood[['Facility', 'Lat', 'Long', 'Flood Depth (meters)']]
        elif 'Exposure' in df_flood.columns:
            df_flood.rename(columns={'Exposure': 'Flood Depth (meters)'}, inplace=True)
            df_flood_values = df_flood[['Facility', 'Lat', 'Long', 'Flood Depth (meters)']]
        else:
            logger.warning("Flood depth column not found in flood analysis output")
            return None, plot_paths
            
        # Collect plot paths
        if flood_res.get('png_paths'):
            plot_paths.extend(flood_res['png_paths'])
            
        return df_flood_values, plot_paths
        
    except Exception as e:
        logger.exception(f"Error in Flood Exposure Analysis: {e}")
        return None, []


def process_water_stress_analysis(facility_csv_path, selected_fields, buffer_size=0.0045):
    """
    Process water stress analysis if selected.
    Args:
    facility_csv_path (str): Path to facility CSV
    selected_fields (list): List of selected hazard types
    buffer_size (float): Buffer size for analysis
    
    Returns:
        tuple: (DataFrame with water stress values, list of plot paths)
    """
    if 'Water Stress' not in selected_fields:
        return None, []
        
    logger.info(f"Integrating Water Stress Analysis with buffer size: {buffer_size}")
    plot_paths = []

    try:
        ws_res = generate_water_stress_analysis(facility_csv_path, buffer_size)
        
        if 'error' in ws_res:
            logger.warning(f"Warning in Water Stress Analysis: {ws_res['error']}")
            return None, []
            
        if not ws_res.get('combined_csv_paths'):
            return None, plot_paths
            
        # Read the water stress analysis CSV
        df_ws = pd.read_csv(ws_res['combined_csv_paths'][0])
        
        # Standardize column names
        rename_map = {'Site': 'Facility', 'latitude': 'Lat', 'longitude': 'Long'}
        for old, new in rename_map.items():
            if old in df_ws.columns and new not in df_ws.columns:
                df_ws.rename(columns={old: new}, inplace=True)
        
        # Handle water stress column variations
        if 'Water Stress Exposure (%)' in df_ws.columns:
            df_ws_values = df_ws[['Facility', 'Lat', 'Long', 'Water Stress Exposure (%)']]
        elif 'bws_06_raw' in df_ws.columns:
            df_ws.rename(columns={'bws_06_raw': 'Water Stress Exposure (%)'}, inplace=True)
            df_ws_values = df_ws[['Facility', 'Lat', 'Long', 'Water Stress Exposure (%)']]
        else:
            logger.warning("Water stress column not found in analysis output")
            return None, plot_paths
            
        # Collect plot paths
        if ws_res.get('png_paths'):
            plot_paths.extend(ws_res['png_paths'])
            
        return df_ws_values, plot_paths
        
    except Exception as e:
        logger.exception(f"Error in Water Stress Analysis: {e}")
        return None, []


def process_sea_level_rise_analysis(facility_csv_path, selected_fields):
    """
    Process sea level rise analysis if selected.
    
    Args:
        facility_csv_path (str): Path to facility CSV
        selected_fields (list): List of selected hazard types
        
    Returns:
        tuple: (DataFrame with SLR values, list of plot paths)
    """
    if 'Sea Level Rise' not in selected_fields:
        return None, []
        
    logger.info("Integrating Sea Level Rise Analysis")
    plot_paths = []
    
    try:
        slr_res = generate_sea_level_rise_analysis(facility_csv_path)
        
        if 'error' in slr_res:
            logger.warning(f"Warning in Sea Level Rise Analysis: {slr_res['error']}")
            return None, []
            
        if not slr_res.get('combined_csv_paths'):
            return None, plot_paths
            
        # Read the SLR analysis CSV
        df_slr = pd.read_csv(slr_res['combined_csv_paths'][0])
        
        # Standardize column names
        rename_map = {'Site': 'Facility', 'Lon': 'Long'}
        for old, new in rename_map.items():
            if old in df_slr.columns and new not in df_slr.columns:
                df_slr.rename(columns={old: new}, inplace=True)
        
        # Standardize sea level rise column names
        years = [2030, 2040, 2050, 2060]
        rename_fields = {
            f"{yr} Sea Level Rise CI 0.5": f"{yr} Sea Level Rise (in meters)"
            for yr in years
        }
        df_slr.rename(columns=rename_fields, inplace=True)
        
        # Rename elevation column if present
        if 'SRTM elevation' in df_slr.columns:
            df_slr.rename(columns={'SRTM elevation': 'Elevation (meter above sea level)'}, inplace=True)
        
        # Get available SLR columns
        slr_cols = ['Elevation (meter above sea level)'] + list(rename_fields.values())
        available_slr_cols = [c for c in slr_cols if c in df_slr.columns]
        
        if not available_slr_cols:
            logger.warning("No SLR columns found in analysis output")
            return None, plot_paths
            
        # Create SLR values dataframe
        slr_values = df_slr[['Facility', 'Lat', 'Long'] + available_slr_cols].copy()
        
        # Collect plot paths
        if slr_res.get('png_paths'):
            plot_paths.extend(slr_res['png_paths'])
            
        return slr_values, plot_paths
        
    except Exception as e:
        logger.exception(f"Error in Sea Level Rise Analysis: {e}")
        return None, []


def process_tropical_cyclone_analysis(facility_csv_path, selected_fields):
    """
    Process tropical cyclone analysis if selected.
    Modified to exclude 200 and 500-year return period columns.
    
    Args:
        facility_csv_path (str): Path to facility CSV
        selected_fields (list): List of selected hazard types
        
    Returns:
        tuple: (DataFrame with TC values, list of plot paths)
    """
    if 'Tropical Cyclones' not in selected_fields:
        return None, []
        
    logger.info("Integrating Tropical Cyclones Analysis")
    plot_paths = []
    
    try:
        tc_res = generate_tropical_cyclone_analysis(facility_csv_path)
        
        if 'error' in tc_res:
            logger.warning(f"Warning in Tropical Cyclones Analysis: {tc_res['error']}")
            return None, []
            
        if not tc_res.get('combined_csv_paths'):
            return None, plot_paths
            
        # Read the TC analysis CSV
        df_tc = pd.read_csv(tc_res['combined_csv_paths'][0])
        
        # Standardize column names
        rename_map = {
            'Facility Name': 'Facility',
            'Latitude': 'Lat',
            'Longitude': 'Long'
        }
        for old, new in rename_map.items():
            if old in df_tc.columns and new not in df_tc.columns:
                df_tc.rename(columns={old: new}, inplace=True)
        
        # Standardize TC column names - EXCLUDE 200 and 500-year periods
        tc_rename = {
            '1-min MSW 10 yr RP': 'Extreme Windspeed 10 year Return Period (km/h)',
            '1-min MSW 20 yr RP': 'Extreme Windspeed 20 year Return Period (km/h)',
            '1-min MSW 50 yr RP': 'Extreme Windspeed 50 year Return Period (km/h)',
            '1-min MSW 100 yr RP': 'Extreme Windspeed 100 year Return Period (km/h)',
            # 200 and 500-year periods are removed
        }
        df_tc.rename(columns=tc_rename, inplace=True)
        
        # Identify TC columns - ONLY include the desired return periods
        tc_cols = [
            'Extreme Windspeed 10 year Return Period (km/h)',
            'Extreme Windspeed 20 year Return Period (km/h)',
            'Extreme Windspeed 50 year Return Period (km/h)',
            'Extreme Windspeed 100 year Return Period (km/h)'
        ]
        
        # Filter columns that exist in the DataFrame
        available_tc_cols = [col for col in tc_cols if col in df_tc.columns]
        
        if not available_tc_cols:
            logger.warning("No TC columns found in analysis output")
            return None, plot_paths
            
        # Create TC values dataframe
        tc_values = df_tc[['Facility', 'Lat', 'Long'] + available_tc_cols].copy()
        
        return tc_values, plot_paths
        
    except Exception as e:
        logger.exception(f"Error in Tropical Cyclones Analysis: {e}")
        return None, []


def process_heat_exposure_analysis(facility_csv_path, selected_fields):
    """
    Process heat exposure analysis if selected.
    
    Args:
        facility_csv_path (str): Path to facility CSV
        selected_fields (list): List of selected hazard types
        
    Returns:
        tuple: (DataFrame with heat values, list of plot paths)
    """
    if 'Heat' not in selected_fields:
        return None, []
        
    logger.info("Integrating Heat Exposure Analysis")
    plot_paths = []
    
    try:
        heat_res = generate_heat_exposure_analysis(facility_csv_path)
        
        if 'error' in heat_res:
            logger.warning(f"Warning in Heat Exposure Analysis: {heat_res['error']}")
            return None, []
            
        if not heat_res.get('combined_csv_paths'):
            return None, plot_paths
            
        # Read the heat analysis CSV
        df_heat = pd.read_csv(heat_res['combined_csv_paths'][0])
        logger.info(f"Heat exposure columns: {df_heat.columns.tolist()}")
        
        # Standardize column names
        rename_map = {'Site': 'Facility', 'latitude': 'Lat', 'longitude': 'Long'}
        for old, new in rename_map.items():
            if old in df_heat.columns and new not in df_heat.columns:
                df_heat.rename(columns={old: new}, inplace=True)
        
        # Standardize heat column names
        heat_cols = []
        
        # Handle original format from heat_exposure_analysis.py
        temp_mapping = {
            'n>30degC_2125': 'Days over 30° Celsius',
            'n>33degC_2125': 'Days over 33° Celsius',
            'n>35degC_2125': 'Days over 35° Celsius'
        }
        
        for old, new in temp_mapping.items():
            if old in df_heat.columns:
                df_heat.rename(columns={old: new}, inplace=True)
                heat_cols.append(new)
            elif new in df_heat.columns:
                heat_cols.append(new)
        
        if not heat_cols:
            logger.warning("No heat columns found in analysis output")
            return None, plot_paths
            
        # Create heat values dataframe
        heat_values = df_heat[['Facility', 'Lat', 'Long'] + heat_cols].copy()
        
        # Collect plot paths
        if heat_res.get('png_paths'):
            plot_paths.extend(heat_res['png_paths'])
            
        return heat_values, plot_paths
        
    except Exception as e:
        logger.exception(f"Error in Heat Exposure Analysis: {e}")
        return None, []


def process_storm_surge_landslide_analysis(df_fac, selected_fields):
    """
    Process storm surge and rainfall induced landslide analyses if selected.
    
    Args:
        df_fac (DataFrame): Facility dataframe with standardized columns
        selected_fields (list): List of selected hazard types
        
    Returns:
        DataFrame: DataFrame with SS/RIL values or None if not applicable
    """
    if not any(h in selected_fields for h in ['Storm Surge', 'Rainfall Induced Landslide']):
        return None
        
    logger.info("Integrating Storm Surge & Rainfall Induced Landslide Analyses")
    
    try:
        # Paths to raster files
        idir = Path(settings.BASE_DIR)/'climate_hazards_analysis'/'static'/'input_files'
        fp_ls = idir/'PH_LandslideHazards_UTM_ProjectNOAH_Unmasked.tif'
        fp_ss = idir/'PH_StormSurge_Advisory4_UTM_ProjectNOAH_Unmasked.tif'
        
        # Check if required raster files exist
        missing_files = []
        if 'Storm Surge' in selected_fields and not os.path.exists(fp_ss):
            missing_files.append(str(fp_ss))
        if 'Rainfall Induced Landslide' in selected_fields and not os.path.exists(fp_ls):
            missing_files.append(str(fp_ls))
            
        if missing_files:
            logger.warning(f"Missing raster files for SS/RIL analysis: {', '.join(missing_files)}")
            return None
        
        # Create a copy of the facility dataframe with lat/long renamed for geopandas
        df_a = df_fac[['Facility', 'Lat', 'Long']].copy()
        df_a.rename(columns={'Lat': 'latitude', 'Long': 'longitude'}, inplace=True)
        
        # Convert coordinates to numeric
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
                logger.warning(f"Warning: {hazard_type} raster not found: {ras}")
                gdf_a[lbl] = 0  # Default value
        
        # Rename columns back for merging
        gdf_a.rename(columns={'latitude': 'Lat', 'longitude': 'Long'}, inplace=True)
        
        # Create final columns
        hazard_cols = []
        if 'Storm Surge' in selected_fields and 'stormsurge_raster' in gdf_a.columns:
            gdf_a.rename(columns={'stormsurge_raster': 'Storm Surge Flood Depth (meters)'}, inplace=True)
            hazard_cols.append('Storm Surge Flood Depth (meters)')
        
        if 'Rainfall Induced Landslide' in selected_fields and 'landslide_raster' in gdf_a.columns:
            gdf_a.rename(columns={'landslide_raster': 'Rainfall Induced Landslide Factor of Safety'}, inplace=True)
            hazard_cols.append('Rainfall Induced Landslide Factor of Safety')
            
        if not hazard_cols:
            return None
            
        return gdf_a[['Facility', 'Lat', 'Long'] + hazard_cols].copy()
        
    except Exception as e:
        logger.exception(f"Error in Storm Surge/Landslide analysis: {e}")
        return None


def process_nan_values(df):
    """
    Replace NaN values with appropriate text based on column type.
    
    Args:
        df (DataFrame): Combined dataframe with all hazard data
        
    Returns:
        DataFrame: Processed dataframe with NaN values replaced
    """
    for col in df.columns:
        if col in ['Facility', 'Lat', 'Long']:
            continue
            
        if 'Sea Level Rise' in col or col == 'Elevation (meter above sea level)':
            df[col] = df[col].apply(
                lambda v: "Little to no effect" if pd.isna(v) else v
            )
        elif 'Extreme Windspeed' in col:
            df[col] = df[col].apply(
                lambda v: "Data not available" if pd.isna(v) else v
            )
        else:
            df[col] = df[col].apply(
                lambda v: "N/A" if pd.isna(v) else v
            )
    
    return df


def generate_climate_hazards_analysis(facility_csv_path=None, selected_fields=None, buffer_size=0.0045):
    """
    Integrates multiple climate hazard analyses into a single output.

    Args:
    facility_csv_path: Path to facility locations CSV (required)
    selected_fields: List of selected hazard types to include
    buffer_size: Buffer size for spatial analysis (default 0.0045)
    
    Returns:
        dict: Results dictionary with paths to combined output and plots
    """
    try:
        # Validate inputs
        if not facility_csv_path or not os.path.exists(facility_csv_path):
            raise ValueError("Facility CSV path is required and must exist.")
            
        if selected_fields is None:
            selected_fields = []
            
        logger.info(f"Starting climate hazards analysis with buffer size: {buffer_size}")
        
        # Define path for final combined output
        input_dir = os.path.join(settings.BASE_DIR, 'climate_hazards_analysis', 'static', 'input_files')
        os.makedirs(input_dir, exist_ok=True)
        
        # Load and standardize facility dataframe
        df_fac = pd.read_csv(facility_csv_path)
        df_fac = standardize_facility_dataframe(df_fac)
        
        # Initialize combined DataFrame with base columns
        combined_df = df_fac[['Facility', 'Lat', 'Long']].copy()
        
        # Track plots for visualization
        all_plot_paths = []
        
        # Process each hazard type with buffer_size parameter
        
        # 1. Flood Exposure Analysis
        flood_values, flood_plots = process_flood_exposure_analysis(
            facility_csv_path, selected_fields, buffer_size
        )
        all_plot_paths.extend(flood_plots)
        
        # 2. Water Stress Analysis
        water_stress_values, ws_plots = process_water_stress_analysis(
            facility_csv_path, selected_fields, buffer_size
        )
        all_plot_paths.extend(ws_plots)
        
        # 3. Other analyses (no buffer size needed for these)
        slr_values, slr_plots = process_sea_level_rise_analysis(
            facility_csv_path, selected_fields
        )
        all_plot_paths.extend(slr_plots)
        
        tc_values, tc_plots = process_tropical_cyclone_analysis(
            facility_csv_path, selected_fields
        )
        all_plot_paths.extend(tc_plots)
        
        heat_values, heat_plots = process_heat_exposure_analysis(
            facility_csv_path, selected_fields
        )
        all_plot_paths.extend(heat_plots)
        
        ss_ril_values = process_storm_surge_landslide_analysis(
            df_fac, selected_fields
        )

        # Merge all hazard data to combined DataFrame in the correct order
        data_frames = [
            (flood_values, "flood exposure"),
            (water_stress_values, "water stress"),
            (slr_values, "sea level rise"),
            (tc_values, "tropical cyclones"),
            (heat_values, "heat exposure"),
            (ss_ril_values, "storm surge/landslide")
        ]
        
        for df_values, name in data_frames:
            if df_values is not None:
                logger.info(f"Merging {name} values with {len(df_values)} rows")
                combined_df = combined_df.merge(
                    df_values, on=['Facility', 'Lat', 'Long'], how='left'
                )

        # Process NaN values
        combined_df = process_nan_values(combined_df)

        # Write combined output CSV with buffer size in filename if not default
        if buffer_size != 0.0045:
            out_csv = os.path.join(input_dir, f'combined_output_buffer_{buffer_size:.4f}.csv')
        else:
            out_csv = os.path.join(input_dir, 'combined_output.csv')
            
        combined_df.to_csv(out_csv, index=False)
        logger.info(f"Saved combined output CSV: {out_csv}")
        
        # Select the main plot for display (prioritizing flood exposure if available)
        main_plot = None
        if all_plot_paths:
            # Prioritize flood exposure plot if available
            flood_plots = [p for p in all_plot_paths if 'flood_exposure' in p.lower()]
            if flood_plots:
                main_plot = flood_plots[0]
            else:
                main_plot = all_plot_paths[0]
        
        return {
            'combined_csv_path': out_csv,
            'plot_path': main_plot,
            'all_plots': all_plot_paths,
            'buffer_size': buffer_size
        }
        
    except Exception as e:
        logger.exception(f"Error in generate_climate_hazards_analysis: {e}")
        return {'error': str(e), 'combined_csv_path': None, 'plot_path': None}
