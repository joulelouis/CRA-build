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
        
        # Create groups for column headers (optional)
        # This is a simplified version, you may want to enhance this
        groups = {
            'Facility Info': 3,  # First 3 columns (Facility, Lat, Long)
            'Climate Hazards': len(columns) - 3  # Rest of the columns
        }
        
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
    # Get selected hazards from the v2 module's session key
    selected_fields = request.session.get('climate_hazards_v2_selected_hazards', [])
    
    # If no hazards are selected, try to get from results
    if not selected_fields:
        # Try to get from results data if available
        results = request.session.get('climate_hazards_v2_results', {})
        if results and 'selected_hazards' in results:
            selected_fields = results.get('selected_hazards', [])
    
    # Initialize a BytesIO buffer for the PDF
    buffer = BytesIO()
    
    # Generate the PDF report
    generate_climate_hazards_report_pdf(buffer, selected_fields)
    
    # Get the PDF content
    pdf = buffer.getvalue()
    buffer.close()
    
    # Create and return the HTTP response with the PDF
    response = HttpResponse(pdf, content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="Climate_Hazard_Exposure_Report_V2.pdf"'
    return response