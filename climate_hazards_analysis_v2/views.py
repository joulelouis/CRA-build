from django.shortcuts import render, redirect
from django.http import JsonResponse
import os
import pandas as pd
import json
from django.conf import settings
from .utils import standardize_facility_dataframe, load_cached_hazard_data, combine_facility_with_hazard_data
import logging

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