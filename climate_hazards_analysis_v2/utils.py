import pandas as pd
import numpy as np
import os
from django.conf import settings
import logging

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
    
    # Standardize facility name column - Include 'Site' as a valid name column
    facility_name_variations = [
        'facility', 'site', 'site name', 
        'facility name', 'facilty name', 'name',
        'asset name'  
    ]
    
     # Find and rename facility name column
    for col in df.columns:
        if col.strip().lower() in facility_name_variations:
            df.rename(columns={col: 'Facility'}, inplace=True)
            break
            
    # Standardize lat/long columns
    coord_mapping = {'latitude': 'Lat', 'longitude': 'Long'}
    for old, new in coord_mapping.items():
        if old.lower() in [c.lower() for c in df.columns]:
            for c in df.columns:
                if c.lower() == old.lower() and new not in df.columns:
                    df.rename(columns={c: new}, inplace=True)
    
    # Validate required columns
    required_cols = ['Facility', 'Lat', 'Long']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.warning(f"Missing required columns in facility CSV: {', '.join(missing)}")
        
        # If Facility column is missing, try to create it from index or another column
        if 'Facility' in missing:
            if 'Name' in df.columns:
                df['Facility'] = df['Name']
            elif 'Site' in df.columns:  # Use Site column if present but not already mapped
                df['Facility'] = df['Site']
            else:
                df['Facility'] = df.index.map(lambda i: f"Facility {i+1}")
                
        # If Lat/Long columns are missing, log error but don't attempt to create them
        if 'Lat' in missing or 'Long' in missing:
            logger.error("Missing coordinates; cannot continue without Lat/Long")
            raise ValueError("Missing coordinates; cannot continue without Lat/Long")
    
    # Convert coordinates to numeric and drop invalid
    df['Lat'] = pd.to_numeric(df['Lat'], errors='coerce')
    df['Long'] = pd.to_numeric(df['Long'], errors='coerce')
    df.dropna(subset=['Lat', 'Long'], inplace=True)
    
    if df.empty:
        raise ValueError("No valid facility locations after processing.")
        
    return df

def validate_shapefile(gdf):
    """Validate that an uploaded shapefile has the expected structure.

    The shapefile must contain point, multipoint, polygon, or multipolygon
    geometries and include a column that can be used for facility names.
    Returns the list of attribute columns (i.e. the columns excluding
    geometry).


    Args:
        gdf (geopandas.GeoDataFrame): The input geodataframe from the shapefile

    Returns:
        list[str]: Attribute column names

    Raises:
        ValueError: If the shapefile has no features, contains geometries other
            than points, multipoints, polygons, or multipolygons, or lacks a
            suitable facility name column.
    """
    if gdf.empty:
        raise ValueError("Shapefile contains no features")

    # Ensure geometries are of supported types
    allowed_geom_types = ["Point", "MultiPoint", "Polygon", "MultiPolygon"]
    if not gdf.geometry.geom_type.isin(allowed_geom_types).all():
        raise ValueError(
            "Shapefile must contain point, multipoint, polygon, or multipolygon geometries"
        )

    attribute_columns = [c for c in gdf.columns if c.lower() != "geometry"]

    facility_name_variations = [
        "facility",
        "site",
        "site name",
        "facility name",
        "facilty name",
        "name",
        "asset name",
    ]

    if not any(col.strip().lower() in facility_name_variations for col in attribute_columns):
        raise ValueError(
            "Shapefile attribute table must include a facility name column (e.g., 'Facility', 'Site', 'Name')."
        )

    return attribute_columns

def load_cached_hazard_data(hazard_type):
    """
    Load pre-computed hazard data from the static files.
    
    Args:
        hazard_type (str): Type of hazard data to load (flood, heat, water_stress, etc.)
        
    Returns:
        dict: Dictionary containing hazard data or None if not found
    """
    try:
        # Look for cached hazard files
        file_paths = {
            'flood': os.path.join(settings.BASE_DIR, 'climate_hazards_analysis', 'static', 'output', 
                                 'flood_exposure_analysis_output.csv'),
            'water_stress': os.path.join(settings.BASE_DIR, 'climate_hazards_analysis', 'static', 'output',
                                         'water_stress_analysis_output.csv'),
            'heat': os.path.join(settings.BASE_DIR, 'climate_hazards_analysis', 'static', 'output',
                                'heat_exposure_analysis_output.csv'),
        }
        
        if hazard_type not in file_paths or not os.path.exists(file_paths[hazard_type]):
            logger.warning(f"No cached data found for {hazard_type}")
            return None
            
        # Load the CSV data
        df = pd.read_csv(file_paths[hazard_type])
        
        # Return as dict to make it easier to work with
        return df.to_dict(orient='records')
        
    except Exception as e:
        logger.exception(f"Error loading cached hazard data for {hazard_type}: {e}")
        return None

def combine_facility_with_hazard_data(facilities, hazard_data_list):
    """
    Enrich facility data with available hazard data based on coordinates.
    
    Args:
        facilities (list): List of facility dictionaries with Lat/Long
        hazard_data_list (list): List of hazard data dictionaries
        
    Returns:
        list: Enriched facility dictionaries with hazard data
    """
    # If no hazard data provided, return original facilities
    if not hazard_data_list:
        return facilities
        
    enriched_facilities = []
    
    # Process each facility
    for facility in facilities:
        enriched_facility = facility.copy()
        
        # Try to match facility with hazard data based on coordinates
        for hazard_data in hazard_data_list:
            if not hazard_data:
                continue
                
            # Find matching facilities in hazard data by coordinates
            matches = [
                h for h in hazard_data
                if abs(h.get('Lat', 0) - facility.get('Lat', 0)) < 0.0001 and
                   abs(h.get('Long', 0) - facility.get('Long', 0)) < 0.0001
            ]
            
            # If matches found, add hazard data to facility
            if matches:
                # Add all fields from the first match except Facility, Lat, Long
                for key, value in matches[0].items():
                    if key not in ['Facility', 'Lat', 'Long']:
                        enriched_facility[key] = value
        
        enriched_facilities.append(enriched_facility)
    
    return enriched_facilities