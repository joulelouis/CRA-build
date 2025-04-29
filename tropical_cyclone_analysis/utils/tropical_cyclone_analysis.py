import pandas as pd
import math
from shapely.wkt import loads
from shapely.geometry import Point
from scipy.spatial import distance
import os
from django.conf import settings


def generate_tropical_cyclone_analysis(facilty_csv_path):
    """
    Performs tropical cyclone analysis for facility locations.
    Modified to work with column names: geometry, 10, 20, 50, 100, 200, 500
    
    Args:
        facilty_csv_path (str): Path to the facility CSV file with locations
        
    Returns:
        dict: Dictionary containing file paths to generated outputs
    """
    # Lists to track generated files
    output_csv_files = []

    try:
        # Look for the file in multiple possible locations
        known_csv = os.path.join(settings.BASE_DIR, 'tropical_cyclone_analysis', 'static', 'input_files', 'climada_output_01.csv')
        alt_path = os.path.join(settings.BASE_DIR, 'climate_hazards_analysis', 'static', 'input_files', 'climada_output_01.csv')
        
        # Check if file exists at primary location, otherwise use alternative
        if os.path.exists(known_csv):
            df_known = pd.read_csv(known_csv)
            print(f"Loaded climada file from primary path with {len(df_known)} rows")
        elif os.path.exists(alt_path):
            df_known = pd.read_csv(alt_path)
            print(f"Loaded climada file from alternative path with {len(df_known)} rows")
        else:
            # If file doesn't exist, create a placeholder result
            print("Warning: climada_output_01.csv not found. Creating placeholder results.")
            df_target = pd.read_csv(facilty_csv_path)
            
            # Ensure required columns exist
            if "Lat" not in df_target.columns or "Long" not in df_target.columns:
                if "Latitude" in df_target.columns and "Longitude" in df_target.columns:
                    df_target.rename(columns={"Latitude": "Lat", "Longitude": "Long"}, inplace=True)
                else:
                    raise ValueError("Target CSV must contain 'Lat'/'Latitude' and 'Long'/'Longitude' columns.")
            
            # Get facility name column
            name_col = None
            for col in ["Facility", "Site", "Name", "FacilityName", "SiteName"]:
                if col in df_target.columns:
                    name_col = col
                    break
                    
            if not name_col:
                name_col = df_target.columns[0]  # Use first column as name if no obvious name column
                
            # Create placeholder results dataframe
            columns = ["Facility Name", "Latitude", "Longitude", 
                       "1-min MSW 10 yr RP", "1-min MSW 20 yr RP", "1-min MSW 50 yr RP", 
                       "1-min MSW 100 yr RP", "1-min MSW 200 yr RP", "1-min MSW 500 yr RP"]
            
            df_result = pd.DataFrame(columns=columns)
            
            # Fill with placeholder data
            for idx, row in df_target.iterrows():
                new_row = {
                    "Facility Name": row[name_col] if name_col in row else f"Facility {idx+1}",
                    "Latitude": row["Lat"],
                    "Longitude": row["Long"],
                    "1-min MSW 10 yr RP": "N/A",
                    "1-min MSW 20 yr RP": "N/A",
                    "1-min MSW 50 yr RP": "N/A",
                    "1-min MSW 100 yr RP": "N/A",
                    "1-min MSW 200 yr RP": "N/A",
                    "1-min MSW 500 yr RP": "N/A"
                }
                df_result = pd.concat([df_result, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save the placeholder results
            output_csv = os.path.join(
                settings.BASE_DIR, 
                'tropical_cyclone_analysis', 
                'static', 
                'input_files', 
                'exposure_results_01.csv'
            )
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df_result.to_csv(output_csv, index=False)
            output_csv_files.append(output_csv)
            
            return {
                "combined_csv_paths": output_csv_files,
            }
            
        # MAP OLD COLUMN NAMES TO NEW ONES
        # This is the key change - map the existing column names to what we need
        column_map = {
            'geometry': 'geometry',  # Keep as is
            '10': 'min_i_1_10',      # Map numeric columns to the expected format
            '20': 'min_i_1_20',
            '50': 'min_i_1_50',
            '100': 'min_i_1_100',
            '200': 'min_i_1_200',
            '500': 'min_i_1_500'
        }
        
        # Check what columns actually exist and create the mapping
        actual_map = {}
        for old_col, new_col in column_map.items():
            if old_col in df_known.columns:
                actual_map[old_col] = new_col
        
        # Rename columns if needed
        if actual_map:
            df_known.rename(columns=actual_map, inplace=True)
            print(f"Renamed columns using mapping: {actual_map}")
        
        # Load target points CSV
        df_target = pd.read_csv(facilty_csv_path)

        # Ensure required columns exist
        if "Lat" not in df_target.columns or "Long" not in df_target.columns:
            if "Latitude" in df_target.columns and "Longitude" in df_target.columns:
                df_target.rename(columns={"Latitude": "Lat", "Longitude": "Long"}, inplace=True)
            else:
                raise ValueError("Target CSV must contain 'Lat'/'Latitude' and 'Long'/'Longitude' columns.")

        # Convert latitude & longitude to Point geometry
        df_target["geometry"] = df_target.apply(lambda row: Point(row["Long"], row["Lat"]), axis=1)

        # Save and reload target file (optional, avoids potential formatting issues)
        output_dir = os.path.join(settings.BASE_DIR, 'tropical_cyclone_analysis', 'static', 'input_files')
        os.makedirs(output_dir, exist_ok=True)
        output_asset_coordinates = os.path.join(output_dir, 'smc_assets_coords.csv')
        df_target.to_csv(output_asset_coordinates, index=False)
        df_target = pd.read_csv(output_asset_coordinates)

        # Parse WKT geometry in known points
        try:
            df_known["geometry"] = df_known["geometry"].apply(lambda x: loads(x).coords[0])  # Convert WKT to (x, y)
        except Exception as e:
            print(f"Warning: Error parsing geometry column: {e}")
            # Try a different approach if the standard parsing fails
            try:
                # Assuming the geometry column has values like POINT(longitude latitude)
                def extract_coords(geom_str):
                    """Extract coordinates from POINT string"""
                    if not isinstance(geom_str, str):
                        return (0, 0)  # Default for non-string values
                    
                    if geom_str.startswith('POINT'):
                        # Extract values from POINT(lon lat) format
                        coords_str = geom_str.replace('POINT(', '').replace(')', '')
                        try:
                            lon, lat = map(float, coords_str.split())
                            return (lon, lat)
                        except:
                            pass
                    
                    # If parsing fails, return a default
                    return (0, 0)
                
                df_known["geometry"] = df_known["geometry"].apply(extract_coords)
                print("Applied custom geometry parsing")
            except Exception as e2:
                print(f"Error with backup geometry parsing: {e2}")
                raise

        # Convert target points to (x, y) tuples
        df_target["geometry"] = df_target.apply(lambda row: (row["Long"], row["Lat"]), axis=1)

        # Function to find the nearest known point for a given target
        def find_nearest(target_point):
            try:
                df_known["distance"] = df_known["geometry"].apply(
                    lambda x: distance.euclidean(target_point, x)
                )
                nearest_row = df_known.loc[df_known["distance"].idxmin()]
                return nearest_row.drop(["geometry", "distance"])  # Drop extra columns
            except Exception as e:
                print(f"Error finding nearest point for {target_point}: {e}")
                # Return empty series with the right columns
                empty_data = {col: None for col in df_known.columns if col not in ["geometry", "distance"]}
                return pd.Series(empty_data)

        # Apply function and **expand results into separate columns**
        df_nearest = df_target["geometry"].apply(find_nearest).apply(pd.Series)

        # Merge nearest values back into target DataFrame
        df_result = df_target.drop(columns=["geometry"]).join(df_nearest)

        # Rename columns
        name_col = None
        for col in ["Facility", "Site", "Name", "FacilityName", "SiteName"]:
            if col in df_result.columns:
                name_col = col
                break
                
        if not name_col:
            name_col = df_result.columns[0]  # Use first column as name if no obvious name column
            
        column_mapping = {
            name_col: "Facility Name",
            "Lat": "Latitude",
            "Long": "Longitude"
        }
        
        # Rename using either the original column names or the renamed ones
        for old_name, new_name in [
            ('min_i_1_10', '1-min MSW 10 yr RP'),
            ('min_i_1_20', '1-min MSW 20 yr RP'),
            ('min_i_1_50', '1-min MSW 50 yr RP'),
            ('min_i_1_100', '1-min MSW 100 yr RP'),
            ('min_i_1_200', '1-min MSW 200 yr RP'),
            ('min_i_1_500', '1-min MSW 500 yr RP'),
            # Also try the original column names
            ('10', '1-min MSW 10 yr RP'),
            ('20', '1-min MSW 20 yr RP'),
            ('50', '1-min MSW 50 yr RP'),
            ('100', '1-min MSW 100 yr RP'),
            ('200', '1-min MSW 200 yr RP'),
            ('500', '1-min MSW 500 yr RP')
        ]:
            if old_name in df_result.columns:
                column_mapping[old_name] = new_name
        
        df_result.rename(columns=column_mapping, inplace=True)
        
        # Remove any duplicate columns
        df_result = df_result.loc[:, ~df_result.columns.duplicated()]
        
        # Make sure we have all required columns
        required_cols = ["Facility Name", "Latitude", "Longitude"]
        msr_cols = [
            "1-min MSW 10 yr RP", "1-min MSW 20 yr RP", "1-min MSW 50 yr RP",
            "1-min MSW 100 yr RP", "1-min MSW 200 yr RP", "1-min MSW 500 yr RP"
        ]
        
        # Ensure all required columns are present
        for col in required_cols + msr_cols:
            if col not in df_result.columns:
                df_result[col] = "N/A"
                
        # Keep only necessary columns
        df_result = df_result[required_cols + msr_cols]

        # Define a helper function that rounds up if not NaN, otherwise returns a fallback string
        def round_up_and_convert(x):
            if pd.isna(x) or x == "N/A":
                return "N/A"
            try:
                return str(int(math.ceil(float(x))))
            except (ValueError, TypeError):
                return "N/A"

        # Apply the helper function to MSW columns
        for col in msr_cols:
            df_result[col] = df_result[col].apply(round_up_and_convert)

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

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in tropical cyclone analysis: {str(e)}")
        
        # Create a minimal output file even if there's an error
        output_dir = os.path.join(settings.BASE_DIR, 'tropical_cyclone_analysis', 'static', 'input_files')
        os.makedirs(output_dir, exist_ok=True)
        
        output_csv = os.path.join(output_dir, 'exposure_results_01.csv')
        
        # Create a simple placeholder file
        df = pd.DataFrame({
            "Facility Name": ["Placeholder due to error"],
            "Latitude": [0],
            "Longitude": [0],
            "1-min MSW 10 yr RP": ["N/A"],
            "1-min MSW 20 yr RP": ["N/A"], 
            "1-min MSW 50 yr RP": ["N/A"],
            "1-min MSW 100 yr RP": ["N/A"],
            "1-min MSW 200 yr RP": ["N/A"],
            "1-min MSW 500 yr RP": ["N/A"]
        })
        df.to_csv(output_csv, index=False)
        output_csv_files.append(output_csv)

    # Return a dictionary with all output file paths.
    return {
        "combined_csv_paths": output_csv_files,
    }