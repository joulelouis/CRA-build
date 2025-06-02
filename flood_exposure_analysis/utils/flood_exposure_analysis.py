import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from pyproj import CRS
from rasterstats import zonal_stats
from django.conf import settings

def generate_flood_exposure_analysis(facility_csv_path, buffer_size=0.0045, flood_thresholds=None):
    """
    Performs flood exposure analysis for facility locations.
    Args:
    facility_csv_path (str): Path to the facility CSV file with locations
    buffer_size (float): Buffer size for analysis (~250m = 0.0045, ~500m = 0.0090, ~750m = 0.0135)
    flood_thresholds (dict): Custom flood threshold parameters with keys:
    - 'little': threshold for "Little to None" category
    - 'low_lower': lower bound for "Low Risk" category
    - 'low_upper': upper bound for "Low Risk" category
    - 'medium_lower': lower bound for "Medium Risk" category
    - 'medium_upper': upper bound for "Medium Risk" category
    - 'high': threshold for "High Risk" category

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
        
        # Load facility locations with proper encoding
        try:
            df_fac = pd.read_csv(facility_csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df_fac = pd.read_csv(facility_csv_path, encoding='latin-1')
            except UnicodeDecodeError:
                df_fac = pd.read_csv(facility_csv_path, encoding='cp1252')
        
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
        
        # Set default flood thresholds if not provided
        if flood_thresholds is None:
            flood_thresholds = {
                'little': 0.3,
                'low_lower': 0.4,
                'low_upper': 1.0,
                'medium_lower': 1.1,
                'medium_upper': 1.5,
                'high': 1.6
            }
        
        # Use the provided buffer size (default ~250 meters for 0.0045)
        buffer = buffer_size
        buffer_meters = int(buffer * 111000)  # Convert to approximate meters for labeling
        
        print(f"Using buffer size: {buffer} degrees (~{buffer_meters}m)")
        print(f"Using flood thresholds: {flood_thresholds}")
        
        # Classify exposure based on flood depth using configurable thresholds
        def determine_exposure(percentile, thresholds):
            if pd.isna(percentile):
                return 'Unknown'
            
            # Convert raster values to actual flood depths
            # Assuming raster values: 1=low flood, 2=medium flood, 3=high flood, etc.
            if percentile == 1:
                depth = 0.3  # Representative value for low flood
            elif percentile == 2:
                depth = 1.0  # Representative value for medium flood
            elif percentile == 3:
                depth = 2.0  # Representative value for high flood
            elif percentile == 0:
                depth = 0.0  # No flood
            else:
                # For other values, use them directly as depth in meters
                depth = float(percentile)
            
            # Classify based on new threshold structure
            if depth <= thresholds['little']:
                return 'Little to None'
            elif thresholds['low_lower'] <= depth <= thresholds['low_upper']:
                return 'Low Risk'
            elif thresholds['medium_lower'] <= depth <= thresholds['medium_upper']:
                return 'Medium Risk'
            elif depth >= thresholds['high']:
                return 'High Risk'
            else:
                # Handle edge cases where depth falls between categories
                if depth < thresholds['low_lower']:
                    return 'Little to None'
                elif depth > thresholds['low_upper'] and depth < thresholds['medium_lower']:
                    return 'Low Risk'  # Assign to lower category for gaps
                elif depth > thresholds['medium_upper'] and depth < thresholds['high']:
                    return 'Medium Risk'  # Assign to lower category for gaps
                else:
                    return 'Unknown'
        
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
            stat['percentile_75'] if stat['percentile_75'] is not None else 0 
            for stat in stats
        ]
        
        # Apply classification and add to GeoDataFrame
        flood_gdf['Exposure'] = flood_gdf['75th Percentile'].apply(
            lambda x: determine_exposure(x, flood_thresholds)
        )
        
        # Rename 'Exposure' to 'Flood Depth (meters)' for the correct column name
        flood_gdf.rename(columns={'Exposure': 'Flood Depth (meters)'}, inplace=True)
        
        # Create a visualization of the flood exposure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a colormap for different exposure categories
        cmap = {
            'Little to None': 'lightgreen',
            'Low Risk': 'green',
            'Medium Risk': 'orange', 
            'High Risk': 'red',
            'Unknown': 'gray'
        }
        
        # Create a point-based GeoDataFrame for visualization
        points_gdf = gpd.GeoDataFrame(
            flood_gdf[['Facility', 'Lat', 'Long', 'Flood Depth (meters)']],
            geometry=gpd.points_from_xy(flood_gdf['Long'], flood_gdf['Lat']),
            crs='EPSG:4326'
        )
        
        # Plot points with colors based on flood risk category
        for category, color in cmap.items():
            category_points = points_gdf[points_gdf['Flood Depth (meters)'] == category]
            if not category_points.empty:
                category_points.plot(ax=ax, color=color, markersize=100, label=category, alpha=0.8)
        
        ax.set_title(f'Flood Exposure for Facility Locations (Buffer: ~{buffer_meters}m)\n'
                    f'Thresholds: Little to None ≤{flood_thresholds["little"]}m, '
                    f'Low {flood_thresholds["low_lower"]}-{flood_thresholds["low_upper"]}m, '
                    f'Medium {flood_thresholds["medium_lower"]}-{flood_thresholds["medium_upper"]}m, '
                    f'High ≥{flood_thresholds["high"]}m')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend(title='Flood Risk Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save the plot with buffer size and threshold info in filename
        plot_filename = f'flood_exposure_plot_buffer_{buffer:.4f}_thresholds.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        output_png_files.append(plot_path)
        
        # Create a summary statistics plot
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Risk category distribution
        risk_counts = flood_gdf['Flood Depth (meters)'].value_counts()
        colors = [cmap.get(cat, 'gray') for cat in risk_counts.index]
        
        ax1.pie(risk_counts.values, labels=risk_counts.index, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Distribution of Flood Risk Categories')
        
        # Risk category bar chart
        bars = ax2.bar(risk_counts.index, risk_counts.values, color=colors)
        ax2.set_title('Facility Count by Flood Risk Category')
        ax2.set_ylabel('Number of Facilities')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the summary statistics plot
        summary_plot_filename = f'flood_exposure_summary_buffer_{buffer:.4f}.png'
        summary_plot_path = os.path.join(output_dir, summary_plot_filename)
        plt.savefig(summary_plot_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        output_png_files.append(summary_plot_path)
        
        # Save the results to CSV with buffer size in filename for sensitivity analysis
        output_filename = f'flood_exposure_analysis_output_buffer_{buffer:.4f}.csv'
        output_csv = os.path.join(output_dir, output_filename)
        
        # Only include relevant columns in the output
        output_data = flood_gdf[['Facility', 'Lat', 'Long', 'Flood Depth (meters)']].copy()
        
        # Add threshold information as metadata in the CSV
        threshold_info = f"# Flood Thresholds Used: Little to None <={flood_thresholds['little']}m, Low {flood_thresholds['low_lower']}-{flood_thresholds['low_upper']}m, Medium {flood_thresholds['medium_lower']}-{flood_thresholds['medium_upper']}m, High >={flood_thresholds['high']}m"

        # Write threshold info as comment and then the data with explicit UTF-8 encoding
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            f.write(threshold_info + '\n')
            output_data.to_csv(f, index=False)
        
        output_csv_files.append(output_csv)
        
        print(f"Flood analysis output saved to: {output_csv}")
        print(f"Generated {len(output_png_files)} visualization files")
        
        # Print summary statistics
        print("\nFlood Risk Summary:")
        for category in ['Little to None', 'Low Risk', 'Medium Risk', 'High Risk', 'Unknown']:
            count = len(output_data[output_data['Flood Depth (meters)'] == category])
            percentage = (count / len(output_data)) * 100 if len(output_data) > 0 else 0
            print(f"  {category}: {count} facilities ({percentage:.1f}%)")
        
        # Return paths to the generated files
        return {
            "combined_csv_paths": output_csv_files,
            "png_paths": output_png_files,
            "buffer_size": buffer_size,
            "buffer_meters": buffer_meters,
            "flood_thresholds": flood_thresholds,
            "summary_stats": {
                "total_facilities": len(output_data),
                "risk_distribution": output_data['Flood Depth (meters)'].value_counts().to_dict()
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}