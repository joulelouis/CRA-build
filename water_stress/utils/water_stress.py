import io
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import os
from django.conf import settings

def generate_water_stress_plot(shapefile_path, dbf_path, shx_path, csv_path, facility_csv_path, dynamic_fields=None, plot_fields=None):
    try:
        print(f"Step 1: Checking input files")

        # Ensure all shapefile components exist
        required_files = [shapefile_path, dbf_path, shx_path]
        missing_files = [file for file in required_files if not os.path.exists(file)]

        if missing_files:
            raise FileNotFoundError(f"Error! Missing required shapefile components: {', '.join(missing_files)}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Error! CSV file {csv_path} not found!")

        if not os.path.exists(facility_csv_path):
            raise FileNotFoundError(f"Error! Facility CSV {facility_csv_path} not found!")

        print("Step 2: Loading shapefile")
        hydrobasins = gpd.read_file(shapefile_path)

        # Check if the shapefile has a CRS
        if hydrobasins.crs is None:
            print("WARNING: Hydrobasins shapefile has no CRS! Assigning EPSG:4326 (WGS84)...")
            hydrobasins.set_crs("EPSG:4326", inplace=True)

        print(f"Shapefile CRS: {hydrobasins.crs}")

        print("Step 3: Loading water risk data CSV")
        water_risk_data = pd.read_csv(csv_path)

        print("Step 4: Checking PFAF_ID in datasets")
        if 'PFAF_ID' not in hydrobasins.columns:
            raise ValueError("Error! PFAF_ID not found in shapefile")
        if 'PFAF_ID' not in water_risk_data.columns:
            raise ValueError("Error! PFAF_ID not found in CSV file")

        print("Step 5: Merging datasets")
        merged_data = hydrobasins.merge(water_risk_data, on='PFAF_ID', suffixes=('_hydro', '_risk'))

        # Step 5a: Save the merged data to a new shapefile
        uploads_dir = os.path.join(settings.BASE_DIR, 'water_stress', 'static', 'input_files')
        os.makedirs(uploads_dir, exist_ok=True)
        
        output_shapefile_path = os.path.join(uploads_dir, 'Aqueduct40_baseline_monthly.shp')
        merged_data.to_file(output_shapefile_path)
        print(f"Merged shapefile saved at {output_shapefile_path}")

        # Step 6: Load the merged shapefile
        print("Step 6: Reloading merged shapefile")
        gdf = gpd.read_file(output_shapefile_path)

        print("Step 7: Loading facility locations CSV")
        facility_locs = pd.read_csv(facility_csv_path)

        #Check if required columns exist
        if 'Long' not in facility_locs.columns or 'Lat' not in facility_locs.columns:
            raise ValueError("Error! Missing 'Longitude' or 'Latitude' columns in facility CSV.")

        print("Step 8: Cleaning facility location data")
        # Convert Longitude & Latitude to numeric
        facility_locs['Long'] = pd.to_numeric(facility_locs['Long'], errors='coerce')
        facility_locs['Lat'] = pd.to_numeric(facility_locs['Lat'], errors='coerce')

        #Drop invalid rows
        facility_locs = facility_locs.dropna(subset=['Long', 'Lat'])

        if facility_locs.empty:
            raise ValueError("Facility locations CSV is empty after cleaning!")

        print("Step 9: Converting facility locations to GeoDataFrame")
        geometry = [Point(xy) for xy in zip(facility_locs['Long'], facility_locs['Lat'])]
        facility_gdf = gpd.GeoDataFrame(facility_locs, geometry=geometry, crs="EPSG:4326")

        if facility_gdf.empty:
            raise ValueError("Facility GeoDataFrame is empty!")

        # Ensure CRS consistency
        facility_gdf.set_geometry('geometry', inplace=True)
        facility_gdf.crs = "EPSG:4326"

        if facility_gdf.crs != gdf.crs:
            print(f"Warning, CRS Mismatch Detected! Converting facility CRS from {facility_gdf.crs} to {gdf.crs}")
            facility_gdf = facility_gdf.to_crs(gdf.crs)

        print("Step 10: Creating square buffers around facilities")
        buffer_size = 0.0009  # ~100 meters radius
        facility_gdf['geometry'] = facility_gdf.geometry.apply(lambda x: box(x.x - buffer_size, x.y - buffer_size, x.x + buffer_size, x.y + buffer_size))

        print("Step 11: Filtering data to focus on specific latitude/longitude bounds")
        min_lat, max_lat = 0, 20
        min_lon, max_lon = 114, 130
        gdf = gdf.cx[min_lon:max_lon, min_lat:max_lat]

        print("Step 12: Performing spatial join")

        # 🔍 Check and fix duplicate indices
        print(f"Checking for duplicate indices in gdf: {gdf.index.duplicated().sum()} duplicates found")
        gdf = gdf.reset_index(drop=True)  # Fix index duplication

        #set dynamic fields if not provided
        if dynamic_fields is None:
            dynamic_fields = ['bws_06_cat', 'bws_06_lab']
        #set plot fields if not provided
        if plot_fields is None:
            plot_fields = ['bws_06_cat']

        #check that each dynamic fields exists in gdf
        for field in dynamic_fields:
            if field not in gdf.columns:
                raise ValueError(f"Field {field} not found in the shapefile data.")

        facility_gdf = gpd.sjoin(facility_gdf, gdf[['geometry']+ dynamic_fields], how='left', predicate='intersects')

        # 🔍 Fix duplicate indices after spatial join
        print(f"Checking for duplicate indices in facility_gdf: {facility_gdf.index.duplicated().sum()} duplicates found")
        facility_gdf = facility_gdf.reset_index(drop=True)

        print("Spatial join completed successfully!")

        #dynamically update facility_locs with joined dynamic fields
        for field in dynamic_fields:
            facility_locs[field] = facility_gdf[field]
       

        output_csv_path = os.path.join(uploads_dir, 'sample_locs_ws.csv')

        print("Step 13: Saving updated facility CSV")
        facility_locs.to_csv(output_csv_path, index=False)

        if os.path.exists(output_csv_path):
            print(f"sample_locs_ws.csv successfully saved at {output_csv_path}")
        else:
            print("ERROR: sample_locs_ws.csv NOT SAVED!")

        print("Step 14: Generating water stress plot")
        fig, ax = plt.subplots(figsize=(12, 8))
        gdf.boundary.plot(ax=ax, linewidth=1, color='black')

        #dynamically plot each selected plot field
        for field in plot_fields:
            if field in gdf.columns: 
                gdf.plot(column=field, ax=ax, legend=True, cmap='OrRd')
            else:
                print(f"Warning: {field} not found in gdf for plotting")
        
        facility_gdf.plot(ax=ax, color='blue', marker='o', markersize=50, alpha=0.5)

        ax.set_title('Water Stress and Facility Locations', fontsize=15)
        ax.set_axis_off()

        plot_path = os.path.join(uploads_dir, 'water_stress_plot.png')
        plt.savefig(plot_path, format='png')
        plt.close(fig)

        if os.path.exists(plot_path):
            print(f"Plot saved successfully at {plot_path}")
        else:
            print("ERROR: Plot NOT SAVED!")

        return plot_path  # Return path to plot for serving dynamically

    except Exception as e:
        print(f"Error in generate_water_stress_plot: {e}")
        return None
