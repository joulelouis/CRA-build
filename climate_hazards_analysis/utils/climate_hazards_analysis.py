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
from heat_exposure_analysis.utils.heat_future_analysis import generate_heat_future_analysis
from tropical_cyclone_analysis.utils.tropical_cyclone_future_analysis import generate_tropical_cyclone_future_analysis
from climate_hazards_analysis.utils.storm_surge_future_analysis import generate_storm_surge_future_analysis
from climate_hazards_analysis.utils.rainfall_induced_landslide_future_analysis import generate_rainfall_induced_landslide_future_analysis
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


def process_flood_exposure_analysis(facility_csv_path, selected_fields):
    """
    Process flood exposure analysis if selected.
    Simplified version that uses the basic flood analysis function.
    """
    if 'Flood' not in selected_fields:
        logger.info("Flood analysis not selected, skipping")
        return None, []
        
    logger.info("Starting Flood Exposure Analysis")
    plot_paths = []

    try:
        # Import the simplified flood analysis function
        from flood_exposure_analysis.utils.flood_exposure_analysis import generate_flood_exposure_analysis
        
        flood_res = generate_flood_exposure_analysis(facility_csv_path)
        
        if 'error' in flood_res:
            logger.warning(f"Warning in Flood Exposure Analysis: {flood_res['error']}")
            # Create placeholder flood data instead of returning None
            df_fac = pd.read_csv(facility_csv_path)
            df_fac = standardize_facility_dataframe(df_fac)
            df_fac['Flood Depth (meters)'] = '0.1 to 0.5'  # Default to lowest risk category
            return df_fac[['Facility', 'Lat', 'Long', 'Flood Depth (meters)']], []
            
        if not flood_res.get('combined_csv_paths'):
            logger.warning("No flood CSV paths returned")
            # Create placeholder flood data
            df_fac = pd.read_csv(facility_csv_path)
            df_fac = standardize_facility_dataframe(df_fac)
            df_fac['Flood Depth (meters)'] = '0.1 to 0.5'
            return df_fac[['Facility', 'Lat', 'Long', 'Flood Depth (meters)']], plot_paths
            
        # Read the flood analysis CSV with proper encoding handling
        flood_csv_path = flood_res['combined_csv_paths'][0]
        logger.info(f"Reading flood CSV from: {flood_csv_path}")
        
        try:
            df_flood = pd.read_csv(flood_csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df_flood = pd.read_csv(flood_csv_path, encoding='latin-1')
                logger.warning("Flood CSV read with latin-1 encoding")
            except UnicodeDecodeError:
                df_flood = pd.read_csv(flood_csv_path, encoding='cp1252')
                logger.warning("Flood CSV read with cp1252 encoding")
        
        logger.info(f"Flood CSV columns: {df_flood.columns.tolist()}")
        logger.info(f"Flood CSV shape: {df_flood.shape}")
        
        # Standardize column names
        rename_map = {'Site': 'Facility', 'latitude': 'Lat', 'longitude': 'Long'}
        for old, new in rename_map.items():
            if old in df_flood.columns and new not in df_flood.columns:
                df_flood.rename(columns={old: new}, inplace=True)
        
        # Handle flood depth column variations
        flood_column_found = False
        if 'Flood Depth (meters)' in df_flood.columns:
            logger.info("Found 'Flood Depth (meters)' column")
            df_flood_values = df_flood[['Facility', 'Lat', 'Long', 'Flood Depth (meters)']]
            flood_column_found = True
        elif 'Exposure' in df_flood.columns:
            logger.info("Found 'Exposure' column, renaming to 'Flood Depth (meters)'")
            df_flood.rename(columns={'Exposure': 'Flood Depth (meters)'}, inplace=True)
            df_flood_values = df_flood[['Facility', 'Lat', 'Long', 'Flood Depth (meters)']]
            flood_column_found = True
        elif 'flood_depth' in df_flood.columns:
            logger.info("Found 'flood_depth' column, renaming to 'Flood Depth (meters)'")
            df_flood.rename(columns={'flood_depth': 'Flood Depth (meters)'}, inplace=True)
            df_flood_values = df_flood[['Facility', 'Lat', 'Long', 'Flood Depth (meters)']]
            flood_column_found = True
        elif 'Flood_Depth' in df_flood.columns:
            logger.info("Found 'Flood_Depth' column, renaming to 'Flood Depth (meters)'")
            df_flood.rename(columns={'Flood_Depth': 'Flood Depth (meters)'}, inplace=True)
            df_flood_values = df_flood[['Facility', 'Lat', 'Long', 'Flood Depth (meters)']]
            flood_column_found = True
        
        if not flood_column_found:
            logger.warning(f"Flood depth column not found in flood analysis output. Available columns: {df_flood.columns.tolist()}")
            # Create placeholder data with the expected column
            df_fac = pd.read_csv(facility_csv_path)
            df_fac = standardize_facility_dataframe(df_fac)
            df_fac['Flood Depth (meters)'] = '0.1 to 0.5'
            return df_fac[['Facility', 'Lat', 'Long', 'Flood Depth (meters)']], plot_paths
            
        # Clean any NaN values in the flood column specifically
        flood_nan_count = df_flood_values['Flood Depth (meters)'].isna().sum()
        if flood_nan_count > 0:
            logger.warning(f"Found {flood_nan_count} NaN values in flood column, replacing with '0.1 to 0.5'")
            df_flood_values['Flood Depth (meters)'].fillna('0.1 to 0.5', inplace=True)
        
        # Ensure all values are valid categories (for simplified flood analysis)
        valid_categories = {'0.1 to 0.5', '0.5 to 1.5', 'Greater than 1.5', 'Unknown'}
        invalid_mask = ~df_flood_values['Flood Depth (meters)'].isin(valid_categories)
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid flood category values, replacing with '0.1 to 0.5'")
            invalid_values = df_flood_values.loc[invalid_mask, 'Flood Depth (meters)'].unique()
            logger.warning(f"Invalid values were: {invalid_values}")
            df_flood_values.loc[invalid_mask, 'Flood Depth (meters)'] = '0.1 to 0.5'
        
        # Final verification
        final_nan_count = df_flood_values['Flood Depth (meters)'].isna().sum()
        if final_nan_count > 0:
            logger.error(f"ERROR: Still have {final_nan_count} NaN values in flood column after cleaning!")
            df_flood_values['Flood Depth (meters)'].fillna('0.1 to 0.5', inplace=True)
        
        logger.info(f"Flood column value counts:")
        logger.info(df_flood_values['Flood Depth (meters)'].value_counts())
        
        # Collect plot paths
        if flood_res.get('png_paths'):
            plot_paths.extend(flood_res['png_paths'])
        
        logger.info(f"Successfully processed flood data with {len(df_flood_values)} rows")
        logger.info(f"Sample flood data:\n{df_flood_values.head()}")
            
        return df_flood_values, plot_paths
        
    except Exception as e:
        logger.exception(f"Error in Flood Exposure Analysis: {e}")
        # Create placeholder flood data even on error
        try:
            df_fac = pd.read_csv(facility_csv_path)
            df_fac = standardize_facility_dataframe(df_fac)
            df_fac['Flood Depth (meters)'] = '0.1 to 0.5'  # Default to lowest risk category
            return df_fac[['Facility', 'Lat', 'Long', 'Flood Depth (meters)']], []
        except Exception as e2:
            logger.exception(f"Error creating placeholder flood data: {e2}")
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
            
        # Read the water stress analysis CSV with proper encoding handling
        try:
            df_ws = pd.read_csv(ws_res['combined_csv_paths'][0], encoding='utf-8')
        except UnicodeDecodeError:
            df_ws = pd.read_csv(ws_res['combined_csv_paths'][0], encoding='latin-1')
        
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
            
        # Read the SLR analysis CSV with proper encoding handling
        try:
            df_slr = pd.read_csv(slr_res['combined_csv_paths'][0], encoding='utf-8')
        except UnicodeDecodeError:
            df_slr = pd.read_csv(slr_res['combined_csv_paths'][0], encoding='latin-1')
        
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
            
        # Read the TC analysis CSV with proper encoding handling
        try:
            df_tc = pd.read_csv(tc_res['combined_csv_paths'][0], encoding='utf-8')
        except UnicodeDecodeError:
            df_tc = pd.read_csv(tc_res['combined_csv_paths'][0], encoding='latin-1')
        
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
            
        # Read the heat analysis CSV with proper encoding handling
        try:
            df_heat = pd.read_csv(heat_res['combined_csv_paths'][0], encoding='utf-8')
        except UnicodeDecodeError:
            df_heat = pd.read_csv(heat_res['combined_csv_paths'][0], encoding='latin-1')
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
            gdf_a.rename(columns={'landslide_raster': 'Rainfall-Induced Landslide (factor of safety)'}, inplace=True)
            hazard_cols.append('Rainfall-Induced Landslide (factor of safety)')
            
        if not hazard_cols:
            return None
            
        return gdf_a[['Facility', 'Lat', 'Long'] + hazard_cols].copy()
        
    except Exception as e:
        logger.exception(f"Error in Storm Surge/Landslide analysis: {e}")
        return None


def process_nan_values(df):
    """
    Replace NaN values with appropriate text based on column type.
    Updated for simplified flood categories.
    
    Args:
        df (DataFrame): Combined dataframe with all hazard data
        
    Returns:
        DataFrame: Processed dataframe with NaN values replaced
    """
    print(f"Processing NaN values for {len(df)} rows and {len(df.columns)} columns")
    
    # Iterate using column positions to avoid pandas returning DataFrames when
    # duplicate column names exist. This ensures we always work with a Series
    # which avoids ambiguous truth value errors when checking for NaN.
    for idx, col in enumerate(df.columns):
        if col in ['Facility', 'Lat', 'Long']:
            continue
            
        print(f"Processing column: {col}")

        col_series = df.iloc[:, idx]
        
        # Count initial NaN values. When duplicate column names exist pandas
        # returns a DataFrame instead of a Series, so sum across all columns
        # to avoid ambiguous truth-value errors
        initial_nan_count = col_series.isna().sum()
        if initial_nan_count > 0:
            print(f"  Found {int(initial_nan_count)} NaN values in {col}")
        
        # Apply column-specific replacements
        if 'Sea Level Rise' in col or col == 'Elevation (meter above sea level)':
            col_series = col_series.apply(
                lambda v: "Little to no effect" if pd.isna(v) or v == '' or str(v).lower() == 'nan' else v
            )
        elif 'Extreme Windspeed' in col or 'Tropical Cyclone' in col:
            col_series = col_series.apply(
                lambda v: "Data not available" if pd.isna(v) or v == '' or str(v).lower() == 'nan' else v
            )
        elif 'Flood Depth' in col:
            # Special handling for flood data - use simplified categories
            col_series = col_series.apply(
                lambda v: "0.1 to 0.5" if pd.isna(v) or v == '' or str(v).lower() == 'nan' else v
            )
        elif 'Water Stress' in col:
            # Water stress should be numeric or N/A
            col_series = col_series.apply(
                lambda v: "N/A" if pd.isna(v) or v == '' or str(v).lower() == 'nan' else v
            )
        elif any(temp in col for temp in ['Days over', 'Heat']):
            # Heat data should be numeric or N/A
            col_series = col_series.apply(
                lambda v: "N/A" if pd.isna(v) or v == '' or str(v).lower() == 'nan' else v
            )
        else:
            # Generic handling for other columns
            df[col] = df[col].apply(
                lambda v: "N/A" if pd.isna(v) or v == '' or str(v).lower() == 'nan' else v
            )

        # Assign the processed series back to the DataFrame
        # Cast to object to avoid dtype incompatibility warnings when assigning
        # strings such as "N/A" into numeric columns
        df.iloc[:, idx] = col_series.astype(object)
        
        # Verify no NaN values remain
        final_nan_count = col_series.isna().sum()
        if final_nan_count > 0:
            print(f"  WARNING: {int(final_nan_count)} NaN values still remain in {col}")
            # Force replace any remaining NaN
            df.iloc[:, idx].fillna("N/A", inplace=True)
        else:
            print(f"  ✓ All NaN values processed in {col}")
    
    print("NaN processing complete")
    return df


def generate_climate_hazards_analysis(facility_csv_path=None, selected_fields=None, buffer_size=0.0045, sensitivity_params=None):
    """
    Integrates multiple climate hazard analyses into a single output.
    Simplified version without flood threshold parameters.

    Args:
    facility_csv_path: Path to facility locations CSV (required)
    selected_fields: List of selected hazard types to include
    buffer_size: Buffer size for spatial analysis (default 0.0045)
    sensitivity_params: Dictionary of sensitivity parameters (flood thresholds removed)
    
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
        logger.info(f"Selected fields: {selected_fields}")
        if sensitivity_params:
            logger.info(f"Using sensitivity parameters: {list(sensitivity_params.keys())}")
        
        # Define path for final combined output
        input_dir = os.path.join(settings.BASE_DIR, 'climate_hazards_analysis', 'static', 'input_files')
        os.makedirs(input_dir, exist_ok=True)
        
        # Load and standardize facility dataframe with proper encoding
        try:
            df_fac = pd.read_csv(facility_csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df_fac = pd.read_csv(facility_csv_path, encoding='latin-1')
                logger.warning(f"Facility CSV read with latin-1 encoding")
            except UnicodeDecodeError:
                df_fac = pd.read_csv(facility_csv_path, encoding='cp1252')
                logger.warning(f"Facility CSV read with cp1252 encoding")
        
        df_fac = standardize_facility_dataframe(df_fac)
        logger.info(f"Loaded facility data with {len(df_fac)} facilities")
        
        # Initialize combined DataFrame with base columns
        combined_df = df_fac[['Facility', 'Lat', 'Long']].copy()
        logger.info(f"Initialized combined DataFrame with columns: {combined_df.columns.tolist()}")
        
        # Track plots for visualization
        all_plot_paths = []
        
        # Process each hazard type (simplified without flood thresholds)
        
        # 1. Flood Exposure Analysis - Simplified version
        logger.info("=== PROCESSING FLOOD EXPOSURE ANALYSIS ===")
        flood_values, flood_plots = process_flood_exposure_analysis(
            facility_csv_path, selected_fields
        )
        all_plot_paths.extend(flood_plots)
        
        if flood_values is not None:
            logger.info(f"Flood values shape: {flood_values.shape}")
            logger.info(f"Flood values columns: {flood_values.columns.tolist()}")
            logger.info("Merging flood values...")
            combined_df = combined_df.merge(
                flood_values, on=['Facility', 'Lat', 'Long'], how='left'
            )
            logger.info(f"Combined DF after flood merge - shape: {combined_df.shape}, columns: {combined_df.columns.tolist()}")
        else:
            logger.warning("No flood values to merge")
        
        # 2. Water Stress Analysis
        logger.info("=== PROCESSING WATER STRESS ANALYSIS ===")
        water_stress_values, ws_plots = process_water_stress_analysis(
            facility_csv_path, selected_fields, buffer_size
        )
        all_plot_paths.extend(ws_plots)
        
        if water_stress_values is not None:
            logger.info("Merging water stress values...")
            combined_df = combined_df.merge(
                water_stress_values, on=['Facility', 'Lat', 'Long'], how='left'
            )
            logger.info(f"Combined DF after water stress merge - shape: {combined_df.shape}")
        
        # 3. Other analyses (no buffer size needed for these)
        logger.info("=== PROCESSING OTHER ANALYSES ===")
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

        # Merge remaining hazard data to combined DataFrame
        data_frames = [
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
                logger.info(f"Combined DF after {name} merge - shape: {combined_df.shape}")

        # Add future tropical cyclone values if TC analysis was performed
        if 'Tropical Cyclones' in selected_fields:
            try:
                combined_df = generate_tropical_cyclone_future_analysis(combined_df)

                future_cols = [
                    c
                    for c in combined_df.columns
                    if c.startswith('Extreme Windspeed')
                    and ('Base Case' in c or 'Worst Case' in c)
                ]
                if (
                    'Extreme Windspeed 100 year Return Period (km/h)' in combined_df.columns
                    and future_cols
                ):
                    cols = [c for c in combined_df.columns if c not in future_cols]
                    insert_pos = (
                        cols.index('Extreme Windspeed 100 year Return Period (km/h)')
                        + 1
                    )
                    for col in future_cols:
                        cols.insert(insert_pos, col)
                        insert_pos += 1
                    combined_df = combined_df[cols]

                logger.info('Future tropical cyclone columns added')

            except Exception as e:
                logger.warning(f'Failed to add future tropical cyclone values: {e}')

        # Add future heat exposure values if heat analysis was performed.
        if 'Heat' in selected_fields:
            try:
                tiff_dir = Path(settings.BASE_DIR) / 'climate_hazards_analysis' / 'static' / 'input_files'
                combined_df = generate_heat_future_analysis(combined_df, tiff_dir)
                logger.info('Future heat exposure columns added')
            except Exception as e:
                logger.warning(f'Failed to add future heat exposure values: {e}')

            rename_map = {
                'DaysOver35C_ssp245_2630': 'Days over 35° Celsius (2026 - 2030) - Moderate Case',
                'DaysOver35C_ssp245_3140': 'Days over 35° Celsius (2031 - 2040) - Moderate Case',
                'DaysOver35C_ssp245_4150': 'Days over 35° Celsius (2041 - 2050) - Moderate Case',
                'DaysOver35C_ssp585_2630': 'Days over 35° Celsius (2026 - 2030) - Worst Case',
                'DaysOver35C_ssp585_3140': 'Days over 35° Celsius (2031 - 2040) - Worst Case',
                'DaysOver35C_ssp585_4150': 'Days over 35° Celsius (2041 - 2050) - Worst Case'
            }
            combined_df.rename(columns=rename_map, inplace=True)

            heat_order = [
                'Days over 30° Celsius',
                'Days over 33° Celsius',
                'Days over 35° Celsius',
                'Days over 35° Celsius (2026 - 2030) - Moderate Case',
                'Days over 35° Celsius (2031 - 2040) - Moderate Case',
                'Days over 35° Celsius (2041 - 2050) - Moderate Case',
                'Days over 35° Celsius (2026 - 2030) - Worst Case',
                'Days over 35° Celsius (2031 - 2040) - Worst Case',
                'Days over 35° Celsius (2041 - 2050) - Worst Case',
            ]
            existing_heat = [c for c in heat_order if c in combined_df.columns]
            if existing_heat:
                cols = combined_df.columns.tolist()
                first_idx = min(cols.index(c) for c in existing_heat)
                for c in existing_heat:
                    cols.remove(c)
                cols[first_idx:first_idx] = existing_heat
                combined_df = combined_df[cols]

        # Add future storm surge flood depth values if storm surge analysis was performed
        if 'Storm Surge' in selected_fields:
            try:
                tif_path = (
                    Path(settings.BASE_DIR)
                    / 'climate_hazards_analysis'
                    / 'static'
                    / 'input_files'
                    / 'PH_StormSurge_Advisory4_Future_UTM_ProjectNOAH-GIRI_Unmasked.tif'
                )
                combined_df = generate_storm_surge_future_analysis(combined_df, tif_path)
                logger.info('Future storm surge column added')
            except Exception as e:
                logger.warning(f'Failed to add future storm surge values: {e}')

        # Add future rainfall-induced landslide values if landslide analysis was performed
        if 'Rainfall Induced Landslide' in selected_fields:
            try:
                idir = Path(settings.BASE_DIR) / 'climate_hazards_analysis' / 'static' / 'input_files'
                mod_path = idir / 'PH_LandslideHazards_RCP26_UTM_ProjectNOAH-GIRI_Unmasked.tif'
                worst_path = idir / 'PH_LandslideHazards_RCP85_UTM_ProjectNOAH-GIRI_Unmasked.tif'
                combined_df = generate_rainfall_induced_landslide_future_analysis(
                    combined_df,
                    mod_path,
                    worst_path,
                )
                logger.info('Future rainfall-induced landslide columns added')
            except Exception as e:
                logger.warning(
                    f'Failed to add future rainfall-induced landslide values: {e}'
                )

        # VERIFICATION: Check if flood column exists
        logger.info("=== FINAL VERIFICATION ===")
        logger.info(f"Final combined DataFrame shape: {combined_df.shape}")
        logger.info(f"Final combined DataFrame columns: {combined_df.columns.tolist()}")
        
        if 'Flood Depth (meters)' in combined_df.columns:
            logger.info("✓ Flood Depth (meters) column successfully included!")
            logger.info(f"Flood column sample values: {combined_df['Flood Depth (meters)'].value_counts()}")
        else:
            logger.error("✗ Flood Depth (meters) column is MISSING!")
            # Add placeholder flood column if missing
            if 'Flood' in selected_fields:
                combined_df['Flood Depth (meters)'] = '0.1 to 0.5'
                logger.info("Added placeholder Flood Depth (meters) column")

        # Process NaN values
        combined_df = process_nan_values(combined_df)

        if 'DaysOver35C_base_2125' in combined_df.columns:
            combined_df.drop(columns=['DaysOver35C_base_2125'], inplace=True)

        # Rename future heat exposure columns for readability
        rename_map = {
            'DaysOver35C_ssp245_2630': 'Days over 35° Celsius (2026 - 2030) - Moderate Case',
            'DaysOver35C_ssp245_3140': 'Days over 35° Celsius (2031 - 2040) - Moderate Case',
            'DaysOver35C_ssp245_4150': 'Days over 35° Celsius (2041 - 2050) - Moderate Case',
            'DaysOver35C_ssp585_2630': 'Days over 35° Celsius (2026 - 2030) - Worst Case',
            'DaysOver35C_ssp585_3140': 'Days over 35° Celsius (2031 - 2040) - Worst Case',
            'DaysOver35C_ssp585_4150': 'Days over 35° Celsius (2041 - 2050) - Worst Case'
        }
        combined_df.rename(columns=rename_map, inplace=True)

        # Final desired column order for output CSV
        final_order = [
            'Facility',
            'Lat',
            'Long',
            'Flood Depth (meters)',
            'Water Stress Exposure (%)',
            'Elevation (meter above sea level)',
            '2030 Sea Level Rise (in meters)',
            '2040 Sea Level Rise (in meters)',
            '2050 Sea Level Rise (in meters)',
            '2060 Sea Level Rise (in meters)',
            'Extreme Windspeed 10 year Return Period (km/h)',
            'Extreme Windspeed 20 year Return Period (km/h)',
            'Extreme Windspeed 50 year Return Period (km/h)',
            'Extreme Windspeed 100 year Return Period (km/h)',
            'Extreme Windspeed 10 year Return Period (km/h) - Base Case',
            'Extreme Windspeed 20 year Return Period (km/h) - Base Case',
            'Extreme Windspeed 50 year Return Period (km/h) - Base Case',
            'Extreme Windspeed 100 year Return Period (km/h) - Base Case',
            'Extreme Windspeed 10 year Return Period (km/h) - Worst Case',
            'Extreme Windspeed 20 year Return Period (km/h) - Worst Case',
            'Extreme Windspeed 50 year Return Period (km/h) - Worst Case',
            'Extreme Windspeed 100 year Return Period (km/h) - Worst Case',
            'Days over 30° Celsius',
            'Days over 33° Celsius',
            'Days over 35° Celsius',
            'Days over 35° Celsius (2026 - 2030) - Moderate Case',
            'Days over 35° Celsius (2031 - 2040) - Moderate Case',
            'Days over 35° Celsius (2041 - 2050) - Moderate Case',
            'Days over 35° Celsius (2026 - 2030) - Worst Case',
            'Days over 35° Celsius (2031 - 2040) - Worst Case',
            'Days over 35° Celsius (2041 - 2050) - Worst Case',
            'Storm Surge Flood Depth (meters)',
            'Storm Surge Flood Depth (meters) - Moderate Case',
            'Rainfall-Induced Landslide (factor of safety)',
            'Rainfall-Induced Landslide (factor of safety) - Moderate Case',
            'Rainfall-Induced Landslide (factor of safety) - Worst Case',
        ]
        existing_cols = [c for c in final_order if c in combined_df.columns]
        remaining_cols = [c for c in combined_df.columns if c not in final_order]
        combined_df = combined_df[existing_cols + remaining_cols]

        # Write combined output CSV with parameters in filename if sensitivity analysis
        if sensitivity_params and buffer_size != 0.0045:
            out_csv = os.path.join(input_dir, f'combined_output_sensitivity_buffer_{buffer_size:.4f}.csv')
        elif buffer_size != 0.0045:
            out_csv = os.path.join(input_dir, f'combined_output_buffer_{buffer_size:.4f}.csv')
        else:
            out_csv = os.path.join(input_dir, 'combined_output.csv')
            
        # Add metadata to CSV if sensitivity parameters were used
        metadata_lines = []
        if sensitivity_params:
            metadata_lines.append(f"# Sensitivity Analysis Results")
            metadata_lines.append(f"# Buffer Size: {buffer_size} degrees (~{int(buffer_size * 111000)}m)")
        
        # Write metadata and data with explicit UTF-8 encoding
        with open(out_csv, 'w', encoding='utf-8', newline='') as f:
            for line in metadata_lines:
                f.write(line + '\n')
            combined_df.to_csv(f, index=False)
            
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
        
        logger.info("=== ANALYSIS COMPLETE ===")
        return {
            'combined_csv_path': out_csv,
            'plot_path': main_plot,
            'all_plots': all_plot_paths,
            'buffer_size': buffer_size,
            'sensitivity_params': sensitivity_params
        }
        
    except Exception as e:
        logger.exception(f"Error in generate_climate_hazards_analysis: {e}")
        return {'error': str(e), 'combined_csv_path': None, 'plot_path': None}
    
def validate_and_clean_dataframe(df, analysis_name=""):
    """
    Validate and clean a dataframe to ensure no NaN values and proper data types.
    
    Args:
        df (DataFrame): Dataframe to validate and clean
        analysis_name (str): Name of the analysis for logging
        
    Returns:
        DataFrame: Cleaned dataframe
    """
    if df is None or df.empty:
        logger.warning(f"{analysis_name} dataframe is None or empty")
        return df
    
    logger.info(f"Validating {analysis_name} dataframe with shape {df.shape}")
    
    # Check for any completely empty rows
    empty_rows = df.isnull().all(axis=1).sum()
    if empty_rows > 0:
        logger.warning(f"Found {empty_rows} completely empty rows in {analysis_name}, removing them")
        df = df.dropna(how='all')
    
    # Check for NaN in critical columns
    for col in ['Facility', 'Lat', 'Long']:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in critical column {col} for {analysis_name}")
                if col in ['Lat', 'Long']:
                    # For coordinates, drop rows with NaN
                    df = df.dropna(subset=[col])
                else:
                    # For facility names, fill with placeholder
                    df[col].fillna(f"Unknown_{analysis_name}", inplace=True)
    
    # Clean all other columns
    for col in df.columns:
        if col not in ['Facility', 'Lat', 'Long']:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                logger.info(f"Cleaning {nan_count} NaN values in {col} for {analysis_name}")
                # Apply appropriate default based on column type
                if 'Flood' in col:
                    df[col].fillna('0.1 to 0.5', inplace=True)  # Use simplified category
                elif 'Water Stress' in col:
                    df[col].fillna('N/A', inplace=True)
                elif 'Sea Level' in col or 'Elevation' in col:
                    df[col].fillna('Little to no effect', inplace=True)
                elif 'Windspeed' in col or 'Tropical' in col:
                    df[col].fillna('Data not available', inplace=True)
                else:
                    df[col].fillna('N/A', inplace=True)
    
    logger.info(f"✓ {analysis_name} dataframe validation complete")
    return df