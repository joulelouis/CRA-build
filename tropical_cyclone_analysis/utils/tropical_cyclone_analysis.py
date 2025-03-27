import pandas as pd
import math
from shapely.wkt import loads
from shapely.geometry import Point
from scipy.spatial import distance
import os
from django.conf import settings


def generate_tropical_cyclone_analysis(facilty_csv_path):

    # Lists to track generated files
    output_csv_files = []

    # Load known points CSV
    known_csv = os.path.join(settings.BASE_DIR, 'tropical_cyclone_analysis', 'static',  'input_files', 'climada_output_01.csv')   # Change to actual file path
    df_known = pd.read_csv(known_csv)

    # Load target points CSV
    # target_csv = "sample_locs.csv"  # Change to actual file path
    df_target = pd.read_csv(facilty_csv_path)

    # Ensure required columns exist
    if "Lat" not in df_target.columns or "Long" not in df_target.columns:
        raise ValueError("Target CSV must contain 'Lat' and 'Long' columns.")

    # Convert latitude & longitude to Point geometry
    df_target["geometry"] = df_target.apply(lambda row: Point(row["Long"], row["Lat"]), axis=1)

    # Save and reload target file (optional, avoids potential formatting issues)
    output_asset_coordinates = os.path.join(settings.BASE_DIR, 'tropical_cyclone_analysis', 'static', 'input_files', 'smc_assets_coords.csv')
    df_target.to_csv(output_asset_coordinates, index=False)
    df_target = pd.read_csv(output_asset_coordinates)

    # Parse WKT geometry in known points
    df_known["geometry"] = df_known["geometry"].apply(lambda x: loads(x).coords[0])  # Convert WKT to (x, y)

    # Convert target points to (x, y) tuples
    df_target["geometry"] = df_target.apply(lambda row: (row["Long"], row["Lat"]), axis=1)

    # Function to find the nearest known point for a given target
    def find_nearest(target_point):
        df_known["distance"] = df_known["geometry"].apply(lambda x: distance.euclidean(target_point, x))
        nearest_row = df_known.loc[df_known["distance"].idxmin()]
        return nearest_row.drop(["geometry", "distance"])  # Drop extra columns

    # Apply function and **expand results into separate columns**
    df_nearest = df_target["geometry"].apply(find_nearest).apply(pd.Series)

    # Merge nearest values back into target DataFrame
    df_result = df_target.drop(columns=["geometry"]).join(df_nearest)


    # Drop the "SBU" column
    # df_result = df_result.drop(columns=["SBU"])

    # Optional: Inspect columns before renaming
    print("Before renaming:", df_result.columns)

    # df_result.columns = ["Facility Name", "SBU", "Latitude", "Longitude", "1-min MSW 10 yr RP", "1-min MSW 20 yr RP", "1-min MSW 50 yr RP", "1-min MSW 100 yr RP"]
    df_result.columns = ["Facility Name", "Latitude", "Longitude", "1-min MSW 10 yr RP", "1-min MSW 20 yr RP", "1-min MSW 50 yr RP", "1-min MSW 100 yr RP", "1-min MSW 200 yr RP", "1-min MSW 500 yr RP"]


    print("After renaming:", df_result.columns)

    # Define a helper function that rounds up if not NaN, otherwise returns a fallback string
    def round_up_and_convert(x):
        if pd.isna(x):
            return "N/A"
        return str(int(math.ceil(x)))

    # List the columns you want to round up
    columns_to_round = [
        "1-min MSW 10 yr RP", 
        "1-min MSW 20 yr RP", 
        "1-min MSW 50 yr RP", 
        "1-min MSW 100 yr RP", 
        "1-min MSW 200 yr RP", 
        "1-min MSW 500 yr RP"
    ]

    # Apply the helper function to each column
    for col in columns_to_round:
        # Ensure the column is converted to float, then apply the function
        df_result[col] = df_result[col].astype(float).apply(round_up_and_convert)

    # Save results to CSV
    output_csv = os.path.join(
        settings.BASE_DIR, 
        'tropical_cyclone_analysis', 
        'static', 
        'input_files', 
        'exposure_results_01.csv'
    )
    df_result.to_csv(output_csv, index=False)

    print("Matching complete! Results saved.")

    output_csv_files.append(output_csv)

    # Return a dictionary with all output file paths.
    return {
        "combined_csv_paths": output_csv_files,
    }