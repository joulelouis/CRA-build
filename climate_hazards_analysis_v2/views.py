from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
import os
from io import BytesIO
import pandas as pd
import json
from django.conf import settings
from .utils import standardize_facility_dataframe, load_cached_hazard_data, combine_facility_with_hazard_data
import logging

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
    This is the third step in the climate hazard analysis workflow.
    """
    # Get facility data and selected hazards from session
    facility_data = request.session.get('climate_hazards_v2_facility_data', [])
    selected_hazards = request.session.get('climate_hazards_v2_selected_hazards', [])
    facility_csv_path = request.session.get('climate_hazards_v2_facility_csv_path')
    
    # Check if we have the necessary data
    if not facility_data or not selected_hazards:
        return redirect('climate_hazards_analysis_v2:select_hazards')
    
    try:
        # Re-use the generate_climate_hazards_analysis function from the original module
        # This function processes the CSV file and generates hazard analysis results
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
                    'Flood',
                    'Water Stress',
                    'Heat',
                    'Sea Level Rise', 
                    'Tropical Cyclones',
                    'Storm Surge',
                    'Rainfall Induced Landslide'
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
                    'Flood',
                    'Water Stress',
                    'Heat',
                    'Sea Level Rise', 
                    'Tropical Cyclones',
                    'Storm Surge',
                    'Rainfall Induced Landslide'
                ],
                'selected_hazards': selected_hazards
            })
        
        # Load the combined CSV file
        df = pd.read_csv(combined_csv_path)
        data = df.to_dict(orient="records")
        columns = df.columns.tolist()
        
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
        
        return render(request, 'climate_hazards_analysis_v2/results.html', context)
        
    except Exception as e:
        logger.exception(f"Error in climate hazards analysis: {str(e)}")
        
        return render(request, 'climate_hazards_analysis_v2/select_hazards.html', {
            'error': f"Error in climate hazards analysis: {str(e)}",
            'facility_count': len(facility_data),
            'hazard_types': [
                'Flood',
                'Water Stress',
                'Heat',
                'Sea Level Rise', 
                'Tropical Cyclones',
                'Storm Surge',
                'Rainfall Induced Landslide'
            ],
            'selected_hazards': selected_hazards
        })
    
def generate_report(request):
    """
    Django view that generates the PDF report and returns it as an HTTP response.
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
    
    Args:
        data (list): List of dictionaries containing facility data with hazard ratings
        selected_hazards (list): List of selected hazard types
        
    Returns:
        dict: Dictionary mapping hazard types to lists of high-risk assets
    """
    high_risk_assets = {}
    
    # Define thresholds for high risk by hazard type
    thresholds = {
        'Flood': {
            'column': 'Flood Depth (meters)',
            'criteria': lambda v: v == 'Greater than 1.5'
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
            'column': 'Storm Surge Hazard Rating',
            'criteria': lambda v: isinstance(v, (int, float)) and v >= 1.5
        },
        'Rainfall Induced Landslide': {
            'column': 'Rainfall Induced Landslide Hazard Rating',
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
                    if isinstance(value, str) and value not in ['N/A', 'Little to no effect', 'Data not available']:
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

    context = {
        'facility_count': len(facility_data),
        'selected_hazards': selected_hazards,
    }

    # Handle form submission
    if request.method == 'POST':
        try:
            # Extract sensitivity parameters from the form
            sensitivity_params = {
                'risk_tolerance': request.POST.get('risk_tolerance', 'medium'),
                'time_horizon': request.POST.get('time_horizon', '2050'),
                'confidence_level': int(request.POST.get('confidence_level', 95)),
                
                # Flood thresholds
                'flood_low_threshold': float(request.POST.get('flood_low_threshold', 0.5)),
                'flood_high_threshold': float(request.POST.get('flood_high_threshold', 1.5)),
                
                # Water stress thresholds
                'water_stress_low': int(request.POST.get('water_stress_low', 10)),
                'water_stress_high': int(request.POST.get('water_stress_high', 30)),
                
                # Heat thresholds
                'heat_30_threshold': int(request.POST.get('heat_30_threshold', 300)),
                'heat_33_threshold': int(request.POST.get('heat_33_threshold', 100)),
                'heat_35_threshold': int(request.POST.get('heat_35_threshold', 30)),
                
                # Wind speed thresholds
                'wind_speed_medium': int(request.POST.get('wind_speed_medium', 119)),
                'wind_speed_high': int(request.POST.get('wind_speed_high', 178)),
                
                # Weighting factors
                'flood_weight': int(request.POST.get('flood_weight', 25)),
                'water_stress_weight': int(request.POST.get('water_stress_weight', 20)),
                'heat_weight': int(request.POST.get('heat_weight', 20)),
                'sea_level_weight': int(request.POST.get('sea_level_weight', 15)),
                'cyclone_weight': int(request.POST.get('cyclone_weight', 15)),
                'other_weight': int(request.POST.get('other_weight', 5)),
                
                # Advanced options
                'enable_monte_carlo': request.POST.get('enable_monte_carlo') == 'on',
                'enable_scenario': request.POST.get('enable_scenario') == 'on',
                'enable_correlation': request.POST.get('enable_correlation') == 'on',
                'iterations': int(request.POST.get('iterations', 1000)),
            }
            
            # Store sensitivity parameters in session
            request.session['climate_hazards_v2_sensitivity_params'] = sensitivity_params
            
            logger.info(f"Sensitivity parameters saved: {sensitivity_params}")
            
            # TODO: Implement sensitivity analysis processing
            # For now, redirect to a placeholder or show success message
            context['success_message'] = "Sensitivity parameters have been saved successfully!"
            
            # Eventually this should redirect to sensitivity results page
            # return redirect('climate_hazards_analysis_v2:sensitivity_results')
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error processing sensitivity parameters: {e}")
            context['error'] = f"Error processing parameters: {str(e)}"
        except Exception as e:
            logger.exception(f"Unexpected error in sensitivity parameters: {e}")
            context['error'] = "An unexpected error occurred while processing parameters."

    # For GET requests or if there was an error, show the form
    return render(request, 'climate_hazards_analysis_v2/sensitivity_parameters.html', context)