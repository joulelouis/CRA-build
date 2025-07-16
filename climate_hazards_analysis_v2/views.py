from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
import os
from io import BytesIO
import pandas as pd
import json
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_GET
from django.utils.decorators import method_decorator
from .utils import standardize_facility_dataframe, load_cached_hazard_data, combine_facility_with_hazard_data
from .error_utils import handle_sensitivity_param_error
import logging
import copy

# Import from climate_hazards_analysis module
from climate_hazards_analysis.utils.climate_hazards_analysis import generate_climate_hazards_analysis
from climate_hazards_analysis.utils.generate_report import generate_climate_hazards_report_pdf
from tropical_cyclone_analysis.utils.tropical_cyclone_analysis import generate_tropical_cyclone_analysis

logger = logging.getLogger(__name__)

def parse_numeric(value, default=0):
    """Convert POST parameter to int or float.

    Returns ``default`` if conversion fails.
    """
    try:
        float_val = float(value)
        return int(float_val) if float_val.is_integer() else float_val
    except (TypeError, ValueError):
        raise ValueError(f"Invalid numeric value: {value}")

def view_map(request):
    """
    Main view for the Climate Hazards Analysis V2 module that displays the map interface
    with upload functionality.
    """
    context = {
        'error': None,
        'success_message': None,
        'uploaded_file_name': request.session.get('climate_hazards_v2_uploaded_filename')
    }
    
    # If a facility CSV or Excel file has been uploaded through the form
    if request.method == 'POST' and request.FILES.get('facility_csv'):
        try:
            # Save the uploaded file
            upload_dir = os.path.join(settings.BASE_DIR, 'climate_hazards_analysis_v2', 'static', 'input_files')
            os.makedirs(upload_dir, exist_ok=True)
            
            file = request.FILES['facility_csv']
            file_path = os.path.join(upload_dir, file.name)
            
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
            
            ext = os.path.splitext(file.name)[1].lower()

            # Process the uploaded file to get facility data
            if ext in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path)
                # Convert to CSV for downstream processing
                csv_path = os.path.splitext(file_path)[0] + '.csv'
                df.to_csv(csv_path, index=False)
                request.session['climate_hazards_v2_facility_csv_path'] = csv_path
            else:
                df = pd.read_csv(file_path)
                request.session['climate_hazards_v2_facility_csv_path'] = file_path
            
            # Standardize column names and validate data
            df = standardize_facility_dataframe(df)
            
            # Store facility data in session for map display
            facility_data = df.to_dict(orient='records')
            
            # Debug: Log the facility data
            logger.info(f"Processed {len(facility_data)} facilities from CSV: {str(facility_data)[:200]}...")
            
            # Explicitly store in session
            request.session['climate_hazards_v2_facility_data'] = facility_data
            request.session.modified = True  # Ensure session is saved
            
            # Add success message to context
            context['success_message'] = f"Successfully loaded {len(facility_data)} facilities from {file.name}"
            
        except Exception as e:
            logger.exception(f"Error processing CSV: {str(e)}")
            context['error'] = f"Error processing CSV: {str(e)}"
    
    # Return the template with context
    return render(request, 'climate_hazards_analysis_v2/main.html', context)

def get_facility_data(request):
    """
    API endpoint to retrieve facility data for the map.
    Returns JSON with facility data including coordinates and available hazard data.
    """
    # Get base facility data from session
    facility_data = request.session.get('climate_hazards_v2_facility_data', [])
    
    if not facility_data:
        return JsonResponse({
            'facilities': []
        })
    
    try:
        # Load cached hazard data
        hazard_data = [
            load_cached_hazard_data('flood'),
            load_cached_hazard_data('water_stress'),
            load_cached_hazard_data('heat')
        ]
        
        # Combine facility data with hazard data
        enriched_facilities = combine_facility_with_hazard_data(facility_data, hazard_data)
        
        return JsonResponse({
            'facilities': enriched_facilities
        })
    except Exception as e:
        logger.exception(f"Error enriching facility data: {e}")
        return JsonResponse({
            'facilities': facility_data,
            'error': str(e)
        })
    

@require_GET
def preview_uploaded_file(request):
    """Return the most recently uploaded facility file for preview."""
    file_path = request.session.get('climate_hazards_v2_facility_csv_path')
    if not file_path or not os.path.exists(file_path):
        return JsonResponse({'error': 'No uploaded file found'}, status=404)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()

    return HttpResponse(content, content_type='text/csv')

def add_facility(request):
    """
    API endpoint to add a new facility from coordinates clicked on the map.
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            lat = data.get('lat')
            lng = data.get('lng')
            name = data.get('name', f"New Facility at {lat:.4f}, {lng:.4f}")
            
            # Get existing facility data or initialize empty list
            facility_data = request.session.get('climate_hazards_v2_facility_data', [])
            
            # Add new facility
            new_facility = {
                'Facility': name,
                'Lat': lat,
                'Long': lng
            }
            facility_data.append(new_facility)
            
            # Update session
            request.session['climate_hazards_v2_facility_data'] = facility_data
            
            return JsonResponse({
                'success': True,
                'facility': new_facility
            })
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=400)
    
    return JsonResponse({
        'success': False,
        'error': 'Only POST method is allowed'
    }, status=405)

def select_hazards(request):
    """
    View for selecting climate/weather hazards for analysis.
    This is the second step in the climate hazard analysis workflow.
    """
    # Get facility data from session
    facility_data = request.session.get('climate_hazards_v2_facility_data', [])
    
    # Define available hazard types
    hazard_types = [
        'Flood',
        'Water Stress',
        'Heat',
        'Sea Level Rise', 
        'Tropical Cyclones',
        'Storm Surge',
        'Rainfall Induced Landslide'
    ]
    
    context = {
        'facility_count': len(facility_data),
        'hazard_types': hazard_types,
        'selected_hazards': request.session.get('climate_hazards_v2_selected_hazards', []),
    }
    
    # Handle form submission
    if request.method == 'POST':
        selected_hazards = request.POST.getlist('hazards')
        request.session['climate_hazards_v2_selected_hazards'] = selected_hazards
        
        # Redirect to results page
        return redirect('climate_hazards_analysis_v2:show_results')
        
    # For GET requests, just display the hazard selection form
    return render(request, 'climate_hazards_analysis_v2/select_hazards.html', context)

def show_results(request):
    """
    View to display climate hazard analysis results.
    Updated to work with simplified flood categories and tropical cyclone integration.
    """

    logger.info("🔥 SHOW_RESULTS FUNCTION CALLED! 🔥")  # ADD THIS LINE
    # Get facility data and selected hazards from session
    facility_data = request.session.get('climate_hazards_v2_facility_data', [])
    selected_hazards = request.session.get('climate_hazards_v2_selected_hazards', [])
    facility_csv_path = request.session.get('climate_hazards_v2_facility_csv_path')
    
    # Check if we have the necessary data
    if not facility_data or not selected_hazards:
        return redirect('climate_hazards_analysis_v2:select_hazards')
    
    try:
        logger.info(f"Starting results processing for {len(facility_data)} facilities")
        logger.info(f"Selected hazards: {selected_hazards}")
        logger.info(f"Facility CSV path: {facility_csv_path}")
        
        # Verify facility CSV file exists
        if not facility_csv_path or not os.path.exists(facility_csv_path):
            logger.error(f"Facility CSV file not found: {facility_csv_path}")
            return render(request, 'climate_hazards_analysis_v2/select_hazards.html', {
                'error': 'Facility CSV file not found. Please upload your facility data again.',
                'facility_count': len(facility_data),
                'hazard_types': [
                    'Flood', 'Water Stress', 'Heat', 'Sea Level Rise', 
                    'Tropical Cyclones', 'Storm Surge', 'Rainfall Induced Landslide'
                ],
                'selected_hazards': selected_hazards
            })
        
        # Re-use the generate_climate_hazards_analysis function from the original module
        logger.info("Calling generate_climate_hazards_analysis...")
        result = generate_climate_hazards_analysis(
            facility_csv_path=facility_csv_path,
            selected_fields=selected_hazards
        )
        
        # Check for errors in the result
        if result is None or 'error' in result:
            error_message = result.get('error', 'Unknown error') if result else 'Analysis failed.'
            logger.error(f"Climate hazards analysis error: {error_message}")
            
            return render(request, 'climate_hazards_analysis_v2/select_hazards.html', {
                'error': error_message,
                'facility_count': len(facility_data),
                'hazard_types': [
                    'Flood', 'Water Stress', 'Heat', 'Sea Level Rise', 
                    'Tropical Cyclones', 'Storm Surge', 'Rainfall Induced Landslide'
                ],
                'selected_hazards': selected_hazards
            })
        
        # Get the combined CSV path and load the data
        combined_csv_path = result.get('combined_csv_path')
        
        if not combined_csv_path or not os.path.exists(combined_csv_path):
            logger.error(f"Combined CSV not found: {combined_csv_path}")
            return render(request, 'climate_hazards_analysis_v2/select_hazards.html', {
                'error': 'Combined analysis output not found.',
                'facility_count': len(facility_data),
                'hazard_types': [
                    'Flood', 'Water Stress', 'Heat', 'Sea Level Rise', 
                    'Tropical Cyclones', 'Storm Surge', 'Rainfall Induced Landslide'
                ],
                'selected_hazards': selected_hazards
            })
        
        # Load the combined CSV file with explicit UTF-8 encoding
        logger.info(f"Loading combined CSV from: {combined_csv_path}")
        try:
            df = pd.read_csv(combined_csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            # Try with different encodings if UTF-8 fails
            try:
                df = pd.read_csv(combined_csv_path, encoding='latin-1')
                logger.warning(f"CSV file {combined_csv_path} read with latin-1 encoding")
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(combined_csv_path, encoding='cp1252')
                    logger.warning(f"CSV file {combined_csv_path} read with cp1252 encoding")
                except UnicodeDecodeError:
                    logger.error(f"Could not read CSV file {combined_csv_path} with any encoding")
                    raise

        # Normalize column names
        df.columns = df.columns.str.strip()
        
        logger.info(f"Loaded CSV with shape: {df.shape}")
        logger.info(f"CSV columns: {df.columns.tolist()}")
        
        # 🚨 EMERGENCY TC DEBUG - Check columns immediately after CSV load
        logger.info("🚨 EMERGENCY TC DEBUG - POST CSV LOAD 🚨")
        logger.info(f"Selected hazards: {selected_hazards}")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Column count: {len(df.columns)}")
        logger.info(f"All columns: {df.columns.tolist()}")

        # Check specifically for TC columns
        tc_expected = [
            'Extreme Windspeed 10 year Return Period (km/h)',
            'Extreme Windspeed 20 year Return Period (km/h)', 
            'Extreme Windspeed 50 year Return Period (km/h)',
            'Extreme Windspeed 100 year Return Period (km/h)'
        ]

        logger.info("🔍 TC COLUMN CHECK:")
        for i, tc_col in enumerate(tc_expected):
            exists = tc_col in df.columns
            logger.info(f"  {i+1}. '{tc_col}' -> {'✅ EXISTS' if exists else '❌ MISSING'}")

        # Check for any TC-related columns
        tc_related = [col for col in df.columns if 'windspeed' in col.lower() or 'extreme' in col.lower() or 'cyclone' in col.lower()]
        logger.info(f"🌀 TC-related columns found: {tc_related}")

        # Check for MSW columns (original TC columns)
        msw_cols = [col for col in df.columns if 'MSW' in col]
        logger.info(f"💨 MSW columns found: {msw_cols}")

        # Count how many TC columns we have
        tc_found = [col for col in tc_expected if col in df.columns]
        logger.info(f"📊 TC SUMMARY: Expected {len(tc_expected)}, Found {len(tc_found)}")
        logger.info(f"📊 TC found columns: {tc_found}")
        logger.info("🚨 END EMERGENCY DEBUG - POST CSV LOAD 🚨")
        
        # CRITICAL: Verify flood column exists and add if missing
        if 'Flood' in selected_hazards and 'Flood Depth (meters)' not in df.columns:
            logger.warning("Flood was selected but Flood Depth (meters) column is missing!")
            df['Flood Depth (meters)'] = '0.1 to 0.5'  # Add placeholder with simplified category
            logger.info("Added placeholder Flood Depth (meters) column")
        
        # Track column count before TC processing
        columns_before_tc = len(df.columns)
        logger.info(f"📊 Columns before TC processing: {columns_before_tc}")
        
        # NEW: Handle Tropical Cyclone analysis if selected
        if 'Tropical Cyclones' in selected_hazards:
            try:
                logger.info("🌀 Processing Tropical Cyclone analysis...")
                from tropical_cyclone_analysis.utils.tropical_cyclone_analysis import generate_tropical_cyclone_analysis
                
                tc_result = generate_tropical_cyclone_analysis(facility_csv_path)
                
                if tc_result and 'combined_csv_paths' in tc_result and tc_result['combined_csv_paths']:
                    tc_csv_path = tc_result['combined_csv_paths'][0]
                    
                    if os.path.exists(tc_csv_path):
                        logger.info(f"Loading tropical cyclone data from: {tc_csv_path}")
                        
                        # Load TC data with same encoding handling
                        try:
                            tc_df = pd.read_csv(tc_csv_path, encoding='utf-8')
                        except UnicodeDecodeError:
                            try:
                                tc_df = pd.read_csv(tc_csv_path, encoding='latin-1')
                            except UnicodeDecodeError:
                                tc_df = pd.read_csv(tc_csv_path, encoding='cp1252')

                        tc_df.columns = tc_df.columns.str.strip()
                        
                        logger.info(f"TC data shape: {tc_df.shape}")
                        logger.info(f"TC columns: {tc_df.columns.tolist()}")
                        logger.info(f"Main data columns: {df.columns.tolist()}")
                        
                        # FIXED: Look for the actual TC column names
                        tc_wind_columns = [col for col in tc_df.columns if 'MSW' in col and 'yr RP' in col]
                        logger.info(f"Found TC wind columns: {tc_wind_columns}")
                        
                        # Try to find matching facility column
                        merge_column = None
                        
                        # Check for facility name columns in both dataframes
                        main_facility_cols = []
                        for col in df.columns:
                            if any(keyword in col.lower() for keyword in ['facility', 'name', 'site']):
                                main_facility_cols.append(col)
                        
                        tc_facility_cols = []
                        for col in tc_df.columns:
                            if any(keyword in col.lower() for keyword in ['facility', 'name', 'site']):
                                tc_facility_cols.append(col)
                        
                        logger.info(f"Main facility columns: {main_facility_cols}")
                        logger.info(f"TC facility columns: {tc_facility_cols}")
                        
                        # Try to find a matching column
                        if main_facility_cols and tc_facility_cols:
                            # Try exact match first
                            for main_col in main_facility_cols:
                                if main_col in tc_facility_cols:
                                    merge_column = main_col
                                    break
                            
                            # If no exact match, try renaming TC column to match main column
                            if not merge_column:
                                merge_column = main_facility_cols[0]  # Use first main facility column
                                tc_merge_column = tc_facility_cols[0]  # Use first TC facility column
                                
                                if tc_merge_column != merge_column:
                                    tc_df = tc_df.rename(columns={tc_merge_column: merge_column})
                                    logger.info(f"Renamed TC column '{tc_merge_column}' to '{merge_column}'")
                        
                        if merge_column and tc_wind_columns:
                            # Select only the columns we need for merging
                            tc_columns_to_merge = [merge_column] + tc_wind_columns
                            tc_df_subset = tc_df[tc_columns_to_merge]
                            
                            column_rename_map = {}
                            for col in tc_df_subset.columns:
                                col_normalized = col.lower().replace('-', ' ').replace('_', ' ')
                                if 'msw' in col_normalized and 'rp' in col_normalized:
                                    if '10' in col_normalized:
                                        column_rename_map[col] = 'Extreme Windspeed 10 year Return Period (km/h)'
                                    elif '20' in col_normalized:
                                        column_rename_map[col] = 'Extreme Windspeed 20 year Return Period (km/h)'
                                    elif '50' in col_normalized:
                                        column_rename_map[col] = 'Extreme Windspeed 50 year Return Period (km/h)'
                                    elif '100' in col_normalized:
                                        column_rename_map[col] = 'Extreme Windspeed 100 year Return Period (km/h)'
                            
                            if column_rename_map:
                                tc_df_subset = tc_df_subset.rename(columns=column_rename_map)
                                logger.info(f"Renamed TC columns: {column_rename_map}")
                            
                            logger.info(f"Merging using column '{merge_column}' with TC columns: {tc_df_subset.columns.tolist()}")
                            
                            # Merge the dataframes
                            merged_df = pd.merge(df, tc_df_subset, on=merge_column, how='left')
                            df = merged_df
                            logger.info(f"Successfully merged tropical cyclone data. New shape: {df.shape}")
                            logger.info(f"New columns after merge: {df.columns.tolist()}")
                            
                            # 🚨 DEBUG: Check TC columns after merge
                            logger.info("🚨 POST-MERGE TC CHECK 🚨")
                            tc_cols_after_merge = [col for col in tc_expected if col in df.columns]
                            logger.info(f"TC columns after merge: {tc_cols_after_merge}")
                            logger.info(f"TC column count after merge: {len(tc_cols_after_merge)}")
                            logger.info("🚨 END POST-MERGE TC CHECK 🚨")
                            
                        else:
                            logger.warning(f"Could not merge TC data. Merge column: {merge_column}, TC wind columns: {tc_wind_columns}")
                            # Add placeholder columns with expected names
                            placeholder_columns = [
                                'Extreme Windspeed 10 year Return Period (km/h)',
                                'Extreme Windspeed 20 year Return Period (km/h)', 
                                'Extreme Windspeed 50 year Return Period (km/h)', 
                                'Extreme Windspeed 100 year Return Period (km/h)'
                            ]
                            for col in placeholder_columns:
                                if col not in df.columns:
                                    df[col] = 85.0
                            logger.info("Added placeholder tropical cyclone columns")
                    else:
                        logger.error(f"Tropical cyclone CSV file not found: {tc_csv_path}")
                        # Add placeholder columns with expected names
                        placeholder_columns = [
                            'Extreme Windspeed 10 year Return Period (km/h)',
                            'Extreme Windspeed 20 year Return Period (km/h)', 
                            'Extreme Windspeed 50 year Return Period (km/h)', 
                            'Extreme Windspeed 100 year Return Period (km/h)'
                        ]
                        for col in placeholder_columns:
                            if col not in df.columns:
                                df[col] = 85.0
                        logger.info("Added placeholder tropical cyclone columns")
                else:
                    logger.error("Tropical cyclone analysis did not return valid results")
                    # Add placeholder columns with expected names
                    placeholder_columns = [
                        'Extreme Windspeed 10 year Return Period (km/h)',
                        'Extreme Windspeed 20 year Return Period (km/h)', 
                        'Extreme Windspeed 50 year Return Period (km/h)', 
                        'Extreme Windspeed 100 year Return Period (km/h)'
                    ]
                    for col in placeholder_columns:
                        if col not in df.columns:
                            df[col] = 85.0
                    logger.info("Added placeholder tropical cyclone columns")
                    
            except Exception as tc_error:
                logger.exception(f"Error in tropical cyclone analysis: {str(tc_error)}")
                # Add placeholder columns with expected names
                placeholder_columns = [
                    'Extreme Windspeed 10 year Return Period (km/h)',
                    'Extreme Windspeed 20 year Return Period (km/h)', 
                    'Extreme Windspeed 50 year Return Period (km/h)', 
                    'Extreme Windspeed 100 year Return Period (km/h)'
                ]
                for col in placeholder_columns:
                    if col not in df.columns:
                        df[col] = 85.0
                logger.info("Added placeholder tropical cyclone columns due to exception")
        
        # Track column count after TC processing
        columns_after_tc = len(df.columns)
        logger.info(f"📊 Columns after TC processing: {columns_after_tc} (change: {columns_after_tc - columns_before_tc})")

        # Clean up potential merge suffixes like _x or _y that may appear
        rename_map = {c: c[:-2] for c in df.columns if c.endswith('_x') or c.endswith('_y')}
        if rename_map:
            logger.info(f"Renaming columns to remove merge suffixes: {rename_map}")
            df.rename(columns=rename_map, inplace=True)
            # Drop any duplicate columns that may remain after renaming
            df = df.loc[:, ~df.columns.duplicated()]

        

        
        # Convert to dict for template
        data = df.to_dict(orient="records")
        columns = df.columns.tolist()
        
        logger.info(f"Final data has {len(data)} rows and {len(columns)} columns")
        logger.info(f"Final columns: {columns}")
        
        # 🚨 EMERGENCY TC DEBUG - Final check before group creation
        logger.info("🚨 EMERGENCY TC DEBUG - PRE GROUP CREATION 🚨")
        tc_final_check = [col for col in tc_expected if col in columns]
        logger.info(f"TC columns in final data: {tc_final_check}")
        logger.info(f"TC column count in final data: {len(tc_final_check)}")
        logger.info("🚨 END EMERGENCY DEBUG - PRE GROUP CREATION 🚨")
        
        # Create detailed column groups for the table header
        groups = {}
        # Base group - Facility Information
        facility_cols = ['Facility', 'Lat', 'Long']
        facility_count = sum(1 for col in facility_cols if col in columns)
        if facility_count > 0:
            groups['Facility Information'] = facility_count
        
        # Create a mapping for each hazard type and its columns
        hazard_columns = {
            'Flood': ['Flood Depth (meters)'],
            'Water Stress': ['Water Stress Exposure (%)'],
            'Sea Level Rise': ['Elevation (meter above sea level)',
                            '2030 Sea Level Rise (in meters)',
                            '2040 Sea Level Rise (in meters)',
                            '2050 Sea Level Rise (in meters)', 
                            '2060 Sea Level Rise (in meters)'],
            'Tropical Cyclones': ['Extreme Windspeed 10 year Return Period (km/h)', 
                                'Extreme Windspeed 20 year Return Period (km/h)', 
                                'Extreme Windspeed 50 year Return Period (km/h)', 
                                'Extreme Windspeed 100 year Return Period (km/h)'],
            'Heat': [
                'Days over 30° Celsius', 'Days over 33° Celsius', 'Days over 35° Celsius',
                'Days over 35° Celsius (2026 - 2030) - Moderate Case',
                'Days over 35° Celsius (2031 - 2040) - Moderate Case',
                'Days over 35° Celsius (2041 - 2050) - Moderate Case',
                'Days over 35° Celsius (2026 - 2030) - Worst Case',
                'Days over 35° Celsius (2031 - 2040) - Worst Case',
                'Days over 35° Celsius (2041 - 2050) - Worst Case'
            ],
            'Storm Surge': [
                'Storm Surge Flood Depth (meters)',
                'Storm Surge Flood Depth (meters) - Worst Case'
            ],
            'Rainfall-Induced Landslide': [
                'Rainfall-Induced Landslide (factor of safety)',
                'Rainfall-Induced Landslide (factor of safety) - Moderate Case',
                'Rainfall-Induced Landslide (factor of safety) - Worst Case'
            ]
        }
        
        # Add column groups for each hazard type that has columns in the data
        for hazard, cols in hazard_columns.items():
            count = sum(1 for col in cols if col in columns)
            logger.info(f"🔍 Group Creation - Checking {hazard}: found {count} columns out of {len(cols)} expected")
            if hazard == 'Tropical Cyclones':
                logger.info(f"🌀 TC specific check: {[col for col in cols if col in columns]}")
            if count > 0:
                groups[hazard] = count
                logger.info(f"✅ Added {hazard} group with {count} columns")
            else:
                logger.warning(f"❌ No columns found for {hazard} group")

        logger.info("=== DEBUG: Column Detection ===")
        logger.info(f"Final columns list: {columns}")
        logger.info(f"Groups created: {groups}")

        # Only count heat-related future scenario columns that start with
        # "Days over 35 Celsius" for Moderate and Worst Case scenarios
        heat_basecase_count = sum(
            1
            for c in columns
            if c.startswith('Days over 35° Celsius') and c.endswith(' - Moderate Case')
        )
        heat_worstcase_count = sum(
            1
            for c in columns
            if c.startswith('Days over 35° Celsius') and c.endswith(' - Worst Case')
        )
        
        heat_baseline_cols = ['Days over 30° Celsius', 'Days over 33° Celsius', 'Days over 35° Celsius']
        heat_baseline_count = sum(1 for c in heat_baseline_cols if c in columns)

        if 'Heat' in groups:
            groups['Heat'] = heat_baseline_count + heat_basecase_count + heat_worstcase_count

        # Tropical Cyclone column counts
        tc_basecase_count = sum(
            1 for c in columns
            if c.endswith(' - Moderate Case') and 'Windspeed' in c
        )
        tc_worstcase_count = sum(
            1 for c in columns
            if c.endswith(' - Worst Case') and 'Windspeed' in c
        )
        tc_baseline_cols = [
            'Extreme Windspeed 10 year Return Period (km/h)',
            'Extreme Windspeed 20 year Return Period (km/h)',
            'Extreme Windspeed 50 year Return Period (km/h)',
            'Extreme Windspeed 100 year Return Period (km/h)'
        ]
        tc_baseline_count = sum(1 for c in tc_baseline_cols if c in columns)

        if 'Tropical Cyclones' in groups:
            groups['Tropical Cyclones'] = (
                tc_baseline_count + tc_basecase_count + tc_worstcase_count
            )

        # Storm Surge column counts
        ss_worstcase_count = sum(
            1 for c in columns
            if c.endswith(' - Worst Case') and 'Storm Surge Flood Depth' in c
        )
        ss_baseline_cols = ['Storm Surge Flood Depth (meters)']
        ss_baseline_count = sum(1 for c in ss_baseline_cols if c in columns)

        if 'Storm Surge' in groups:
            groups['Storm Surge'] = ss_baseline_count + ss_worstcase_count

        # Rainfall-Induced Landslide column counts
        ls_moderatecase_count = sum(
            1 for c in columns
            if c.endswith(' - Moderate Case') and 'Landslide' in c
        )
        ls_worstcase_count = sum(
            1 for c in columns
            if c.endswith(' - Worst Case') and 'Landslide' in c
        )
        ls_baseline_cols = ['Rainfall-Induced Landslide (factor of safety)']
        ls_baseline_count = sum(1 for c in ls_baseline_cols if c in columns)

        if 'Rainfall-Induced Landslide' in groups:
            groups['Rainfall-Induced Landslide'] = (
                ls_baseline_count + ls_moderatecase_count + ls_worstcase_count
            )


        
        if 'Flood' in selected_hazards:
            flood_col_exists = 'Flood Depth (meters)' in columns
            logger.info(f"'Flood Depth (meters)' in columns: {flood_col_exists}")
        
        # Enhanced TC Debug
        if 'Tropical Cyclones' in selected_hazards:
            tc_expected = ['Extreme Windspeed 10 year Return Period (km/h)', 
                          'Extreme Windspeed 20 year Return Period (km/h)', 
                          'Extreme Windspeed 50 year Return Period (km/h)', 
                          'Extreme Windspeed 100 year Return Period (km/h)']
            tc_found = [col for col in tc_expected if col in columns]
            logger.info(f"'Tropical Cyclones' expected columns: {tc_expected}")
            logger.info(f"'Tropical Cyclones' found columns: {tc_found}")
            logger.info(f"'Tropical Cyclones' found count: {len(tc_found)}")
        
        # Verify specific hazard groups were added if selected
        if 'Flood' in selected_hazards:
            if 'Flood' in groups:
                logger.info(f"✓ Flood group successfully added to table headers")
            else:
                logger.error("✗ Flood group missing from table headers!")
                        
        if 'Tropical Cyclones' in selected_hazards:
            if 'Tropical Cyclones' in groups:  # ← Correct key name!
                logger.info(f"✅ Tropical Cyclones group successfully added to table headers")
            else:
                logger.error("❌ Tropical Cyclones group missing from table headers!")
                # Additional detailed debug
                logger.error(f"❌ Groups dict: {groups}")
                logger.error(f"❌ Selected hazards: {selected_hazards}")
                logger.error(f"❌ TC columns expected: {hazard_columns['Tropical Cyclones']}")
                tc_debug_found = [col for col in hazard_columns['Tropical Cyclones'] if col in columns]
                logger.error(f"❌ TC columns actually found: {tc_debug_found}")
        
        # Get the paths to any generated plots
        plot_path = result.get('plot_path')
        all_plots = result.get('all_plots', [])
        
        # Store analysis results in session for potential reuse
        request.session['climate_hazards_v2_results'] = {
            'data': data,
            'columns': columns,
            'plot_path': plot_path if plot_path else None,
            'all_plots': all_plots
        }

        # Preserve a baseline copy of the results the first time they are
        # generated so we can restore them later if needed
        if 'climate_hazards_v2_baseline_results' not in request.session:
            request.session['climate_hazards_v2_baseline_results'] = copy.deepcopy(
                request.session['climate_hazards_v2_results']
            )
        
        # Prepare context for the template
        context = {
            'data': data,
            'columns': columns,
            'groups': groups,
            'plot_path': plot_path,
            'all_plots': all_plots,
            'selected_hazards': selected_hazards,
            'heat_basecase_count': heat_basecase_count,
            'heat_worstcase_count': heat_worstcase_count,
            'heat_baseline_count': heat_baseline_count,
            'tc_basecase_count': tc_basecase_count,
            'tc_worstcase_count': tc_worstcase_count,
            'tc_baseline_count': tc_baseline_count,
            'ss_baseline_count': ss_baseline_count,
            'ss_worstcase_count': ss_worstcase_count,
            'ls_baseline_count': ls_baseline_count,
            'ls_moderatecase_count': ls_moderatecase_count,
            'ls_worstcase_count': ls_worstcase_count,
            'success_message': f"Successfully analyzed {len(data)} facilities for {len(selected_hazards)} hazard types."
        }
        
        logger.info("Rendering results template...")
        return render(request, 'climate_hazards_analysis_v2/results.html', context)
        
    except Exception as e:
        logger.exception(f"Error in climate hazards analysis: {str(e)}")
        
        return render(request, 'climate_hazards_analysis_v2/select_hazards.html', {
            'error': f"Error in climate hazards analysis: {str(e)}",
            'facility_count': len(facility_data),
            'hazard_types': [
                'Flood', 'Water Stress', 'Heat', 'Sea Level Rise', 
                'Tropical Cyclones', 'Storm Surge', 'Rainfall Induced Landslide'
            ],
            'selected_hazards': selected_hazards
        })
    
def generate_report(request):
    """
    Django view that generates the PDF report and returns it as an HTTP response.
    Updated to handle simplified flood categories.
    """
    # Get selected hazards and facility data
    selected_fields = request.session.get('climate_hazards_v2_selected_hazards', [])
    facility_data = request.session.get('climate_hazards_v2_facility_data', [])
    
    # Try to get results data if available
    results_data = request.session.get('climate_hazards_v2_results', {})
    if not selected_fields and results_data:
        selected_fields = results_data.get('selected_hazards', [])
    
    # Get the analysis data needed for high-risk asset identification
    analysis_data = []
    if results_data and 'data' in results_data:
        analysis_data = results_data.get('data', [])
    elif 'combined_csv_path' in request.session:
        # Load from CSV if available
        csv_path = request.session.get('combined_csv_path')
        if csv_path and os.path.exists(csv_path):
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                analysis_data = df.to_dict(orient='records')
            except Exception as e:
                print(f"Error loading data for report: {e}")
    
    # Identify high-risk assets for each hazard type
    high_risk_assets = identify_high_risk_assets(analysis_data, selected_fields)
    
    # Initialize a BytesIO buffer for the PDF
    buffer = BytesIO()
    
    # Generate the PDF report with dynamic high-risk assets
    generate_climate_hazards_report_pdf(buffer, selected_fields, high_risk_assets)
    
    # Get the PDF content
    pdf = buffer.getvalue()
    buffer.close()
    
    # Create and return the HTTP response with the PDF
    response = HttpResponse(pdf, content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="Climate_Hazard_Exposure_Report_V2.pdf"'
    return response

def identify_high_risk_assets(data, selected_hazards):
    """
    Identify assets with high risk ratings for each hazard type.
    Updated to handle simplified flood categories.
    
    Args:
        data (list): List of dictionaries containing facility data with hazard ratings
        selected_hazards (list): List of selected hazard types
        
    Returns:
        dict: Dictionary mapping hazard types to lists of high-risk assets
    """
    high_risk_assets = {}
    
    # Define thresholds for high risk by hazard type (updated for simplified flood categories)
    thresholds = {
        'Flood': {
            'column': 'Flood Depth (meters)',
            'criteria': lambda v: v in ['Greater than 1.5', 'High Risk']  # Support both old and new categories
        },
        'Water Stress': {
            'column': 'Water Stress Exposure (%)',
            'criteria': lambda v: isinstance(v, (int, float)) and v > 30
        },
        'Heat': {
            'columns': {
                'Days over 30° Celsius': lambda v: isinstance(v, (int, float)) and v >= 300,
                'Days over 33° Celsius': lambda v: isinstance(v, (int, float)) and v >= 100,
                'Days over 35° Celsius': lambda v: isinstance(v, (int, float)) and v >= 30
            },
            'criteria': lambda row, cols: any(cols[col](row.get(col)) for col in cols if col in row)
        },
        'Sea Level Rise': {
            'column': '2050 Sea Level Rise (in meters)',
            'criteria': lambda v: v != 'Little to no effect' and isinstance(v, (int, float)) and v > 0.5
        },
        'Tropical Cyclones': {
            'column': 'Extreme Windspeed 100 year Return Period (km/h)',
            'criteria': lambda v: v != 'Data not available' and isinstance(v, (int, float)) and v >= 178
        },
        'Storm Surge': {
            'column': 'Storm Surge Flood Depth (meters)',
            'criteria': lambda v: isinstance(v, (int, float)) and v >= 1.5
        },
        'Rainfall Induced Landslide': {
            'column': 'Rainfall-Induced Landslide (factor of safety)',
            'criteria': lambda v: isinstance(v, (int, float)) and v < 1
        }
    }
    
    # Process each selected hazard
    for hazard in selected_hazards:
        if hazard not in thresholds:
            continue
        
        high_risk_assets[hazard] = []
        threshold = thresholds[hazard]
        
        for asset in data:
            facility_name = asset.get('Facility', 'Unknown')
            
            # Special case for Heat which has multiple columns
            if hazard == 'Heat' and 'columns' in threshold:
                if threshold['criteria'](asset, threshold['columns']):
                    high_risk_assets[hazard].append({
                        'name': facility_name,
                        'lat': asset.get('Lat'),
                        'lng': asset.get('Long')
                    })
                continue
            
            # Standard case with single column
            column = threshold.get('column')
            if column in asset:
                value = asset[column]
                try:
                    # Try to convert string values to numbers if possible
                    if isinstance(value, str) and value not in ['N/A', 'Little to no effect', 'Data not available', 
                                                               '0.1 to 0.5', '0.5 to 1.5', 'Greater than 1.5', 'Unknown']:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    
                    if threshold['criteria'](value):
                        high_risk_assets[hazard].append({
                            'name': facility_name,
                            'lat': asset.get('Lat'),
                            'lng': asset.get('Long')
                        })
                except (TypeError, ValueError):
                    continue
    
    return high_risk_assets

def sensitivity_parameters(request):
    """
    View for setting sensitivity parameters for climate hazard analysis.
    This is the fourth step in the climate hazard analysis workflow.
    """
    # Get facility data and selected hazards from session
    facility_data = request.session.get('climate_hazards_v2_facility_data', [])
    selected_hazards = request.session.get('climate_hazards_v2_selected_hazards', [])

    # Check if we have the necessary data from previous steps
    if not facility_data or not selected_hazards:
        return redirect('climate_hazards_analysis_v2:select_hazards')

    # Extract Asset Archetypes from the facility data
    asset_archetypes = []
    facility_csv_path = request.session.get('climate_hazards_v2_facility_csv_path')

    if facility_csv_path and os.path.exists(facility_csv_path):
        try:
            # Read the CSV file to get asset archetypes
            try:
                df = pd.read_csv(facility_csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(facility_csv_path, encoding='latin-1')
                except UnicodeDecodeError:
                    df = pd.read_csv(facility_csv_path, encoding='cp1252')
            
            # Look for Asset Archetype column with various naming conventions
            archetype_column = None
            possible_names = [
                'Asset Archetype', 'asset archetype', 'AssetArchetype', 'assetarchetype',
                'Archetype', 'archetype', 'Asset Type', 'asset type', 'AssetType', 'assettype',
                'Type', 'type', 'Category', 'category', 'Asset Category', 'asset category'
            ]
            
            for col_name in possible_names:
                if col_name in df.columns:
                    archetype_column = col_name
                    break
            
            if archetype_column:
                # Get unique asset archetypes, removing NaN values and sorting
                unique_archetypes = df[archetype_column].dropna().unique()
                unique_archetypes = sorted([str(arch).strip() for arch in unique_archetypes if str(arch).strip()])
                
                # Create numbered list
                asset_archetypes = [
                    {'number': i + 1, 'name': archetype} 
                    for i, archetype in enumerate(unique_archetypes)
                ]
                
                logger.info(f"Found {len(asset_archetypes)} asset archetypes in column '{archetype_column}'")
            else:
                logger.warning("No Asset Archetype column found in the facility CSV")
                asset_archetypes = [{'number': 1, 'name': 'Default Archetype'}]
                
        except Exception as e:
            logger.exception(f"Error reading asset archetypes from CSV: {e}")
            asset_archetypes = [{'number': 1, 'name': 'Default Archetype'}]
    else:
        asset_archetypes = [{'number': 1, 'name': 'Default Archetype'}]

    context = {
        'facility_count': len(facility_data),
        'selected_hazards': selected_hazards,
        'asset_archetypes': asset_archetypes,
    }

    # Handle form submission
    if request.method == 'POST':
        try:
            # Get the selected archetype
            selected_archetype = request.POST.get('selected_archetype', '').strip()
            
            # Extract Water Stress sensitivity parameters from the form
            water_stress_params = {
                'water_stress_low': parse_numeric(request.POST.get('water_stress_low', 10)),
                'water_stress_high': parse_numeric(request.POST.get('water_stress_high', 31)),
                'water_stress_not_material': int(request.POST.get('water_stress_not_material', 0)),
            }

            # Extract Heat sensitivity parameters from the form
            heat_params = {
                'heat_low': parse_numeric(request.POST.get('heat_low', 10)),
                'heat_high': parse_numeric(request.POST.get('heat_high', 45)),
                'heat_not_material': int(request.POST.get('heat_not_material', 0)),
            }

            # Extract Flood sensitivity parameters from the form
            flood_params = {
                'flood_low': parse_numeric(request.POST.get('flood_low', 0.5)),
                'flood_high': parse_numeric(request.POST.get('flood_high', 1.5)),
                'flood_not_material': int(request.POST.get('flood_not_material', 0)),
            }

            # Extract Tropical Cyclone sensitivity parameters from the form
            tropical_cyclone_params = {
                'tropical_cyclone_low': parse_numeric(request.POST.get('tropical_cyclone_low', 119)),
                'tropical_cyclone_high': parse_numeric(request.POST.get('tropical_cyclone_high', 178)),
                'tropical_cyclone_not_material': int(request.POST.get('tropical_cyclone_not_material', 0)),
            }

            # Extract Storm Surge sensitivity parameters from the form
            storm_surge_params = {
                'storm_surge_low': parse_numeric(request.POST.get('storm_surge_low', 0.5)),
                'storm_surge_high': parse_numeric(request.POST.get('storm_surge_high', 1.5)),
                'storm_surge_not_material': int(request.POST.get('storm_surge_not_material', 0)),
            }
            
            logger.info(f"Water Stress parameters received: {water_stress_params}")
            logger.info(f"Heat parameters received: {heat_params}")
            
            # Get existing archetype parameters from session
            archetype_params = request.session.get('climate_hazards_v2_archetype_params', {})
            
            # Check if archetype parameters were submitted through the form
            collected_archetype_params = {}
            for key in request.POST.keys():
                if key.startswith('archetype_params['):
                    # Parse the archetype_params[archetype_name][param_name] format
                    import re
                    match = re.match(r'archetype_params\[([^\]]+)\]\[([^\]]+)\]', key)
                    if match:
                        archetype_name, param_name = match.groups()
                        if archetype_name not in collected_archetype_params:
                            collected_archetype_params[archetype_name] = {}
                        collected_archetype_params[archetype_name][param_name] = parse_numeric(request.POST.get(key))

            # Combine parameters
            combined_params = {
                **water_stress_params,
                **heat_params,
                **flood_params,
                **tropical_cyclone_params,
                **storm_surge_params,
            }
            
            # Use collected parameters if they exist, otherwise use current form values
            if collected_archetype_params:
                archetype_params.update(collected_archetype_params)
                logger.info(f"Updated archetype parameters from form submission: {collected_archetype_params}")
            elif selected_archetype:
                archetype_params[selected_archetype] = combined_params
                logger.info(f"Saved parameters for archetype '{selected_archetype}': {combined_params}")
            else:
                archetype_params['_default'] = combined_params
                logger.info(f"Saved default parameters: {combined_params}")
            
            # Update session with archetype parameters
            request.session['climate_hazards_v2_archetype_params'] = archetype_params
            request.session.modified = True
            
            # Check if Apply Parameters button was clicked
            submit_type = request.POST.get('submit_type', '')
            if submit_type == 'apply-parameters-btn':
                logger.info("Redirecting to sensitivity results")
                return redirect('climate_hazards_analysis_v2:sensitivity_results')
            else:
                context['success_message'] = "Sensitivity parameters saved!"
            
        except (ValueError, TypeError) as e:
            handle_sensitivity_param_error(context, e)
        except Exception as e:
            logger.exception(f"Unexpected error in sensitivity parameter processing: {e}")
            context['error'] = "An unexpected error occurred while processing parameters."

    # For GET requests or if there was an error, show the form
    return render(request, 'climate_hazards_analysis_v2/sensitivity_parameters.html', context)


def sensitivity_results(request):
    """
    View to display climate hazard analysis results with archetype-specific sensitivity parameters.
    This is step 5 in the climate hazard analysis workflow.
    """
    # Get facility data and selected hazards from session
    facility_data = request.session.get('climate_hazards_v2_facility_data', [])
    selected_hazards = request.session.get('climate_hazards_v2_selected_hazards', [])
    facility_csv_path = request.session.get('climate_hazards_v2_facility_csv_path')
    archetype_params = request.session.get('climate_hazards_v2_archetype_params', {})
    
    # Check if we have the necessary data
    if not facility_data or not selected_hazards:
        return redirect('climate_hazards_analysis_v2:select_hazards')
    
    if not archetype_params:
        return render(request, 'climate_hazards_analysis_v2/sensitivity_parameters.html', {
            'error': 'No sensitivity parameters found. Please set sensitivity parameters first.',
            'facility_count': len(facility_data),
            'selected_hazards': selected_hazards,
            'asset_archetypes': []
        })
    
    try:
        logger.info(f"Starting sensitivity results processing for {len(facility_data)} facilities")
        logger.info(f"Selected hazards: {selected_hazards}")
        logger.info(f"Archetype parameters: {archetype_params}")
        
        # Get the original analysis results from step 3
        original_results = request.session.get('climate_hazards_v2_results')
        if not original_results:
            logger.error("Original analysis results not found in session")
            return redirect('climate_hazards_analysis_v2:show_results')
        
        # Create a copy of the original data for sensitivity analysis
        sensitivity_data = copy.deepcopy(original_results['data'])
        columns = original_results['columns'].copy()
        
        # Remove Lat and Long columns from the sensitivity results
        columns_to_remove = ['Lat', 'Long']
        for col_to_remove in columns_to_remove:
            if col_to_remove in columns:
                columns.remove(col_to_remove)
        
        # Remove Lat and Long from each data row
        for row in sensitivity_data:
            for col_to_remove in columns_to_remove:
                if col_to_remove in row:
                    del row[col_to_remove]
        
        logger.info(f"Removed Lat/Long columns. Remaining columns: {columns}")
        logger.info(f"Loaded original data with {len(sensitivity_data)} rows")
        
        # Load the facility CSV to get archetype information
        archetype_mapping = {}
        if facility_csv_path and os.path.exists(facility_csv_path):
            try:
                df = pd.read_csv(facility_csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(facility_csv_path, encoding='latin-1')
                except UnicodeDecodeError:
                    df = pd.read_csv(facility_csv_path, encoding='cp1252')
            
            logger.info(f"Loaded CSV with columns: {df.columns.tolist()}")
            
            # Find archetype column
            archetype_column = None
            possible_names = [
                'Asset Archetype', 'asset archetype', 'AssetArchetype', 'assetarchetype',
                'Archetype', 'archetype', 'Asset Type', 'asset type', 'AssetType', 'assettype',
                'Type', 'type', 'Category', 'category', 'Asset Category', 'asset category'
            ]
            
            for col_name in possible_names:
                if col_name in df.columns:
                    archetype_column = col_name
                    break
            
            if archetype_column:
                logger.info(f"Found archetype column: '{archetype_column}'")
                # Create mapping from facility name to archetype
                for _, row in df.iterrows():
                    # Try multiple facility name columns
                    facility_name = None
                    for name_col in ['Facility', 'Site', 'Name', 'Asset Name']:
                        if name_col in df.columns and pd.notna(row.get(name_col)):
                            facility_name = str(row.get(name_col)).strip()
                            break
                    
                    archetype = str(row.get(archetype_column, '')).strip()
                    if facility_name and archetype and archetype.lower() not in ['', 'nan', 'none']:
                        archetype_mapping[facility_name] = archetype
                        logger.info(f"Mapped '{facility_name}' → '{archetype}'")
                
                logger.info(f"Created archetype mapping for {len(archetype_mapping)} facilities")
                logger.info(f"Archetype mapping: {archetype_mapping}")
            else:
                logger.warning(f"No archetype column found. Available columns: {df.columns.tolist()}")
        else:
            logger.warning("Facility CSV file not found or doesn't exist")
        
        # Apply archetype-specific Water Stress sensitivity parameters
        if 'Water Stress' in selected_hazards and 'Water Stress Exposure (%)' in columns:
            # Debug: log all facility names from sensitivity data
            sensitivity_facility_names = [row.get('Facility', '') for row in sensitivity_data]
            logger.info(f"Facility names in sensitivity data: {sensitivity_facility_names}")
            logger.info(f"Facility names in archetype mapping: {list(archetype_mapping.keys())}")
            
            for row in sensitivity_data:
                facility_name = row.get('Facility', '').strip()
                archetype = archetype_mapping.get(facility_name)
                
                if not archetype:
                    # If no exact match, try to find a partial match
                    for mapped_name, mapped_archetype in archetype_mapping.items():
                        # Try different matching strategies
                        if (facility_name.lower() in mapped_name.lower() or 
                            mapped_name.lower() in facility_name.lower() or
                            facility_name.lower().replace(' ', '') == mapped_name.lower().replace(' ', '')):
                            archetype = mapped_archetype
                            logger.info(f"Used partial match: '{facility_name}' → '{mapped_name}' → '{archetype}'")
                            break
                
                if not archetype:
                    # Try removing common prefixes/suffixes and matching
                    clean_facility_name = facility_name.lower().strip()
                    for mapped_name, mapped_archetype in archetype_mapping.items():
                        clean_mapped_name = mapped_name.lower().strip()
                        # Check if core facility names match (ignoring common words)
                        facility_words = set(clean_facility_name.split())
                        mapped_words = set(clean_mapped_name.split())
                        # If they share at least 2 words, consider it a match
                        if len(facility_words.intersection(mapped_words)) >= 2:
                            archetype = mapped_archetype
                            logger.info(f"Used word-based match: '{facility_name}' → '{mapped_name}' → '{archetype}'")
                            break
                
                if not archetype:
                    archetype = 'Default'
                    logger.warning(f"No archetype found for facility '{facility_name}', using 'Default'")
                else:
                    logger.info(f"Assigned archetype '{archetype}' to facility '{facility_name}'")
                
                # Get parameters for this archetype (or default)
                params = archetype_params.get(archetype, archetype_params.get('_default', {
                    'water_stress_low': 10,
                    'water_stress_high': 31,
                    'storm_surge_low': 0.5,
                    'storm_surge_high': 1.5,
                }))
                
                # Store the archetype and parameters used for this facility (for template access)
                row['Asset Archetype'] = archetype
                row['WS_Low_Threshold'] = params['water_stress_low']
                row['WS_High_Threshold'] = params['water_stress_high']
                row['SS_Low_Threshold'] = params.get('storm_surge_low', 0.5)
                row['SS_High_Threshold'] = params.get('storm_surge_high', 1.5)
                
                logger.info(f"Applied thresholds for '{facility_name}' ({archetype}): Low<{params['water_stress_low']}%, High>{params['water_stress_high']}%")
        
        # Add new columns to the columns list and reorder to put Asset Archetype as 2nd column
        new_columns = ['Asset Archetype', 'WS_Low_Threshold', 'WS_High_Threshold',
                       'SS_Low_Threshold', 'SS_High_Threshold']
        
        # Reorder columns to put Asset Archetype as 2nd column
        if 'Asset Archetype' not in columns:
            # Create new ordered columns list
            ordered_columns = []
            
            # Add Facility first
            if 'Facility' in columns:
                ordered_columns.append('Facility')
            
            # Add Asset Archetype second
            ordered_columns.append('Asset Archetype')
            
            # Add remaining columns (excluding the ones we're repositioning)
            for col in columns:
                if col not in ['Facility', 'Asset Archetype', 'WS_Low_Threshold', 'WS_High_Threshold', 'SS_Low_Threshold', 'SS_High_Threshold']:
                    ordered_columns.append(col)
            
            # Add threshold columns at the end
            if 'WS_Low_Threshold' not in columns:
                ordered_columns.append('WS_Low_Threshold')
            if 'WS_High_Threshold' not in columns:
                ordered_columns.append('WS_High_Threshold')
            if 'SS_Low_Threshold' not in columns:
                ordered_columns.append('SS_Low_Threshold')
            if 'SS_High_Threshold' not in columns:
                ordered_columns.append('SS_High_Threshold')
            
            # Update the columns list
            columns = ordered_columns
            
            logger.info(f"Reordered columns for sensitivity results: {columns}")
        
        # CRITICAL: Reorder the actual row data to match the column order
        # This ensures that when the template iterates through row.items(), 
        # the values appear in the correct column positions
        reordered_data = []
        for row in sensitivity_data:
            ordered_row = {}
            for column in columns:
                if column in row:
                    ordered_row[column] = row[column]
                else:
                    ordered_row[column] = 'N/A'  # Default for missing columns
            reordered_data.append(ordered_row)
        
        sensitivity_data = reordered_data
        logger.info(f"Reordered row data to match column order. Sample row keys: {list(sensitivity_data[0].keys()) if sensitivity_data else 'No data'}")
        
        logger.info(f"Applied sensitivity parameters to {len(sensitivity_data)} facilities")
        
        # Create detailed column groups for the table header (same as original but with new columns)
        groups = {}
        # Base group - Facility Information
        facility_cols = ['Facility', 'Lat', 'Long', 'Asset Archetype',
                        'WS Low Threshold', 'WS High Threshold',
                        'SS Low Threshold', 'SS High Threshold']
        facility_count = sum(1 for col in facility_cols if col in columns)
        if facility_count > 0:
            groups['Facility Information'] = facility_count
        
        # Create a mapping for each hazard type and its columns (excluding separate risk level column)
        hazard_columns = {
            'Flood': ['Flood Depth (meters)'],
            'Water Stress': ['Water Stress Exposure (%)'],  # Only the original exposure column
            'Sea Level Rise': ['Elevation (meter above sea level)', 
                            '2030 Sea Level Rise (in meters)', 
                            '2040 Sea Level Rise (in meters)', 
                            '2050 Sea Level Rise (in meters)', 
                            '2060 Sea Level Rise (in meters)'],
            'Tropical Cyclones': ['Extreme Windspeed 10 year Return Period (km/h)', 
                                'Extreme Windspeed 20 year Return Period (km/h)', 
                                'Extreme Windspeed 50 year Return Period (km/h)', 
                                'Extreme Windspeed 100 year Return Period (km/h)'],
            'Heat': ['Days over 30° Celsius', 'Days over 33° Celsius', 'Days over 35° Celsius',
                     'Days over 35° Celsius (2026 - 2030)', 'Days over 35° Celsius (2031 - 2040)', 'Days over 35° Celsius (2041 - 2050)',
                     'Days over 35° Celsius (2026 - 2030)', 'Days over 35° Celsius (2031 - 2040)', 'Days over 35° Celsius (2041 - 2050)'],
            'Storm Surge': [
                'Storm Surge Flood Depth (meters)',
                'Storm Surge Flood Depth (meters) - Worst Case'
            ],
            'Rainfall-Induced Landslide': [
                'Rainfall-Induced Landslide (factor of safety)',
                'Rainfall-Induced Landslide (factor of safety) - Moderate Case',
                'Rainfall-Induced Landslide (factor of safety) - Worst Case'
            ]
        }
        
        # Add column groups for each hazard type that has columns in the data
        for hazard, cols in hazard_columns.items():
            count = sum(1 for col in cols if col in columns)
            if count > 0:
                groups[hazard] = count
                logger.info(f"Added {hazard} group with {count} columns")
        
        # Store sensitivity results in session
        request.session['climate_hazards_v2_sensitivity_results'] = {
            'data': sensitivity_data,
            'columns': columns,
            'archetype_params': archetype_params
        }
        
        # Prepare context for the template
        context = {
            'data': sensitivity_data,
            'columns': columns,
            'groups': groups,
            'selected_hazards': selected_hazards,
            'archetype_params': archetype_params,
            'is_sensitivity_results': True,  # Flag to indicate this is sensitivity results
            'success_message': f"Successfully applied Water Stress sensitivity parameters to {len(sensitivity_data)} facilities. Water Stress Exposure (%) values now use archetype-specific color coding."
        }
        
        logger.info("Rendering sensitivity results template...")
        return render(request, 'climate_hazards_analysis_v2/sensitivity_results.html', context)
        
    except Exception as e:
        logger.exception(f"Error in sensitivity results: {str(e)}")
        
        return render(request, 'climate_hazards_analysis_v2/sensitivity_parameters.html', {
            'error': f"Error generating sensitivity results: {str(e)}",
            'facility_count': len(facility_data),
            'selected_hazards': selected_hazards,
            'asset_archetypes': []
        })
    
@require_http_methods(["POST"])
def save_table_changes(request):
    """
    Handle AJAX requests to save changes made to the Adjust Magnitude with Local Conditions table.
    Updates the session data and optionally saves to CSV file.
    """
    try:
        # Parse the JSON data from the request
        data = json.loads(request.body)
        changes = data.get('changes', [])
        
        if not changes:
            return JsonResponse({'success': False, 'error': 'No changes provided'})
        
        logger.info(f"Processing {len(changes)} table changes")
        
        # Get current asset exposure results data from session
        results_data = request.session.get('climate_hazards_v2_results', {})
        
        if not results_data or 'data' not in results_data:
            return JsonResponse({'success': False, 'error': 'No asset exposure data found in session'})
        
        # Get the current data
        table_data = results_data['data']
        
        # Apply changes to the data
        for change in changes:
            row_index = change['rowIndex']
            column_name = change['column']
            new_value = change['newValue']
            facility_name = change['facilityName']
            
            # Find the correct row in the data (match by facility name for safety)
            target_row = None
            for i, row in enumerate(table_data):
                if i == row_index and row.get('Facility') == facility_name:
                    target_row = row
                    break
            
            if target_row is None:
                logger.warning(f"Could not find row {row_index} for facility {facility_name}")
                continue
            
            # Convert value to appropriate type
            converted_value = convert_table_value(new_value, column_name)
            
            # Update the value
            target_row[column_name] = converted_value
            logger.info(f"Updated {facility_name} - {column_name}: {converted_value}")
        
        # Update session data for asset exposure results
        results_data['data'] = table_data
        request.session['climate_hazards_v2_results'] = results_data
        
        # Optionally save to CSV file for persistence
        try:
            save_updated_data_to_csv(table_data, request)
        except Exception as csv_error:
            logger.warning(f"Failed to save to CSV: {csv_error}")
            # Don't fail the request if CSV save fails
        
        return JsonResponse({
            'success': True, 
            'message': f'Successfully updated {len(changes)} values',
            'changes_applied': len(changes)
        })
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request: {e}")
        return JsonResponse({'success': False, 'error': 'Invalid JSON data'})
    
    except Exception as e:
        logger.exception(f"Error saving table changes: {e}")
        return JsonResponse({'success': False, 'error': str(e)})


@require_http_methods(["POST"])
def reset_table_data(request):
    """Restore the asset exposure results to the original baseline."""
    baseline = request.session.get('climate_hazards_v2_baseline_results')
    if not baseline:
        return JsonResponse({'success': False, 'error': 'No baseline data available'})

    request.session['climate_hazards_v2_results'] = copy.deepcopy(baseline)
    return JsonResponse({'success': True})

def convert_table_value(value, column_name):
    """
    Convert string value to appropriate type based on column name.
    """
    if value in ['', 'N/A', 'Data not available']:
        return value
    
    # Numeric columns
    numeric_columns = [
        'Flood Depth (meters)',
        'Water Stress Exposure (%)',
        'Days over 30° Celsius',
        'Days over 33° Celsius',
        'Days over 35° Celsius',
        '2030 Sea Level Rise (in meters)',
        '2040 Sea Level Rise (in meters)',
        '2050 Sea Level Rise (in meters)',
        '2060 Sea Level Rise (in meters)',
        'Extreme Windspeed 10 year Return Period (km/h)',
        'Extreme Windspeed 20 year Return Period (km/h)',
        'Extreme Windspeed 50 year Return Period (km/h)',
        'Extreme Windspeed 100 year Return Period (km/h)',
        'Storm Surge Flood Depth (meters)',
        'Storm Surge Flood Depth (meters) - Worst Case'
        'Rainfall-Induced Landslide (factor of safety)',
        'Elevation (meter above sea level)'
    ]
    
    if column_name in numeric_columns:
        try:
            # Convert to float, then to int if it's a whole number
            float_val = float(value)
            if float_val.is_integer():
                return int(float_val)
            return float_val
        except (ValueError, TypeError):
            logger.warning(f"Could not convert '{value}' to number for column '{column_name}'")
            return value
    
    return str(value)


def save_updated_data_to_csv(table_data, request):
    """
    Save the updated table data to a CSV file for persistence.
    """
    try:
        # Create DataFrame from the updated data
        df = pd.DataFrame(table_data)
        
        # Generate filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"asset_exposure_updated_{timestamp}.csv"
        
        # Save to a designated directory (adjust path as needed)
        output_dir = os.path.join('media', 'climate_hazards_v2', 'updated_data')
        os.makedirs(output_dir, exist_ok=True)
        
        file_path = os.path.join(output_dir, filename)
        df.to_csv(file_path, index=False)
        
        # Store the updated file path in session for reference
        request.session['climate_hazards_v2_asset_exposure_updated_csv_path'] = file_path
        
        logger.info(f"Updated data saved to: {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving updated data to CSV: {e}")
        raise