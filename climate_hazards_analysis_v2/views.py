from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
import os
from io import BytesIO
import pandas as pd
import json
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from .utils import standardize_facility_dataframe, load_cached_hazard_data, combine_facility_with_hazard_data
import logging
import copy

# Import from climate_hazards_analysis module
from climate_hazards_analysis.utils.climate_hazards_analysis import generate_climate_hazards_analysis
from climate_hazards_analysis.utils.generate_report import generate_climate_hazards_report_pdf

logger = logging.getLogger(__name__)

def view_map(request):
    """
    Main view for the Climate Hazards Analysis V2 module that displays the map interface
    with upload functionality.
    """
    context = {
        'error': None,
        'success_message': None,
    }
    
    # If a facility CSV has been uploaded through the form
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
            
            # Store file path in session
            request.session['climate_hazards_v2_facility_csv_path'] = file_path
            
            # Process the CSV to get facility data
            df = pd.read_csv(file_path)
            
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
    Updated to work with simplified flood categories.
    """
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
        
        logger.info(f"Loaded CSV with shape: {df.shape}")
        logger.info(f"CSV columns: {df.columns.tolist()}")
        
        # CRITICAL: Verify flood column exists and add if missing
        if 'Flood' in selected_hazards and 'Flood Depth (meters)' not in df.columns:
            logger.warning("Flood was selected but Flood Depth (meters) column is missing!")
            df['Flood Depth (meters)'] = '0.1 to 0.5'  # Add placeholder with simplified category
            logger.info("Added placeholder Flood Depth (meters) column")
        
        # Convert to dict for template
        data = df.to_dict(orient="records")
        columns = df.columns.tolist()
        
        logger.info(f"Final data has {len(data)} rows and {len(columns)} columns")
        
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
            'Tropical Cyclone': ['Extreme Windspeed 10 year Return Period (km/h)', 
                                'Extreme Windspeed 20 year Return Period (km/h)', 
                                'Extreme Windspeed 50 year Return Period (km/h)', 
                                'Extreme Windspeed 100 year Return Period (km/h)'],
            'Heat': ['Days over 30° Celsius', 'Days over 33° Celsius', 'Days over 35° Celsius'],
            'Storm Surge': ['Storm Surge Flood Depth (meters)'],
            'Rainfall-Induced Landslide': ['Rainfall Induced Landslide Factor of Safety']
        }
        
        # Add column groups for each hazard type that has columns in the data
        for hazard, cols in hazard_columns.items():
            count = sum(1 for col in cols if col in columns)
            if count > 0:
                groups[hazard] = count
                logger.info(f"Added {hazard} group with {count} columns")
        
        # Verify flood group was added if flood was selected
        if 'Flood' in selected_hazards:
            if 'Flood' in groups:
                logger.info(f"✓ Flood group successfully added to table headers")
            else:
                logger.error("✗ Flood group missing from table headers!")
        
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
        
        # Prepare context for the template
        context = {
            'data': data,
            'columns': columns,
            'groups': groups,
            'plot_path': plot_path,
            'all_plots': all_plots,
            'selected_hazards': selected_hazards,
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
            'column': 'Rainfall Induced Landslide Factor of Safety',
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
    View for setting Water Stress sensitivity parameters for climate hazard analysis.
    This is the fourth step in the climate hazard analysis workflow.
    Now supports archetype-specific parameter configuration for Water Stress only.
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
                'water_stress_low': int(request.POST.get('water_stress_low', 10)),
                'water_stress_medium_lower': int(request.POST.get('water_stress_medium_lower', 11)),
                'water_stress_medium_upper': int(request.POST.get('water_stress_medium_upper', 30)),
                'water_stress_high': int(request.POST.get('water_stress_high', 31)),
            }
            
            logger.info(f"Water Stress parameters received: {water_stress_params}")
            
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
                        collected_archetype_params[archetype_name][param_name] = int(request.POST.get(key))
            
            # Use collected parameters if they exist, otherwise use current form values
            if collected_archetype_params:
                archetype_params.update(collected_archetype_params)
                logger.info(f"Updated archetype parameters from form submission: {collected_archetype_params}")
            elif selected_archetype:
                # Store parameters for specific archetype
                archetype_params[selected_archetype] = water_stress_params
                logger.info(f"Saved Water Stress parameters for archetype '{selected_archetype}': {water_stress_params}")
            else:
                # Store as default parameters for all archetypes
                archetype_params['_default'] = water_stress_params
                logger.info(f"Saved default Water Stress parameters: {water_stress_params}")
            
            # Update session with archetype parameters
            request.session['climate_hazards_v2_archetype_params'] = archetype_params
            request.session.modified = True
            
            # Check if Apply Parameters button was clicked
            submit_type = request.POST.get('submit_type', '')
            if submit_type == 'apply-parameters-btn':
                logger.info("Redirecting to sensitivity results")
                return redirect('climate_hazards_analysis_v2:sensitivity_results')
            else:
                context['success_message'] = f"Water Stress parameters saved!"
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error processing Water Stress sensitivity parameters: {e}")
            context['error'] = f"Error processing parameters: {str(e)}"
        except Exception as e:
            logger.exception(f"Unexpected error in Water Stress sensitivity parameters: {e}")
            context['error'] = "An unexpected error occurred while processing parameters."

    # For GET requests or if there was an error, show the form
    return render(request, 'climate_hazards_analysis_v2/sensitivity_parameters.html', context)


def sensitivity_results(request):
    """
    View to display climate hazard analysis results with archetype-specific Water Stress sensitivity parameters.
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
            'error': 'No sensitivity parameters found. Please set Water Stress parameters first.',
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
                    'water_stress_medium_lower': 11,
                    'water_stress_medium_upper': 30,
                    'water_stress_high': 31
                }))
                
                # Store the archetype and parameters used for this facility (for template access)
                row['Asset Archetype'] = archetype
                row['WS_Low_Threshold'] = params['water_stress_low']
                row['WS_High_Threshold'] = params['water_stress_high']
                
                logger.info(f"Applied thresholds for '{facility_name}' ({archetype}): Low<{params['water_stress_low']}%, High>{params['water_stress_high']}%")
        
        # Add new columns to the columns list and reorder to put Asset Archetype as 2nd column
        new_columns = ['Asset Archetype', 'WS_Low_Threshold', 'WS_High_Threshold']
        
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
                if col not in ['Facility', 'Asset Archetype', 'WS_Low_Threshold', 'WS_High_Threshold']:
                    ordered_columns.append(col)
            
            # Add threshold columns at the end
            if 'WS_Low_Threshold' not in columns:
                ordered_columns.append('WS_Low_Threshold')
            if 'WS_High_Threshold' not in columns:
                ordered_columns.append('WS_High_Threshold')
            
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
        facility_cols = ['Facility', 'Lat', 'Long', 'Asset Archetype', 'WS Low Threshold', 'WS High Threshold']
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
            'Tropical Cyclone': ['Extreme Windspeed 10 year Return Period (km/h)', 
                                'Extreme Windspeed 20 year Return Period (km/h)', 
                                'Extreme Windspeed 50 year Return Period (km/h)', 
                                'Extreme Windspeed 100 year Return Period (km/h)'],
            'Heat': ['Days over 30° Celsius', 'Days over 33° Celsius', 'Days over 35° Celsius'],
            'Storm Surge': ['Storm Surge Flood Depth (meters)'],
            'Rainfall-Induced Landslide': ['Rainfall Induced Landslide Factor of Safety']
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
    Handle AJAX requests to save changes made to the Asset Exposure Results table.
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
        'Rainfall Induced Landslide Factor of Safety',
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