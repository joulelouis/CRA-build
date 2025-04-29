import os
import pandas as pd
from io import BytesIO
from django.http import HttpResponse
import datetime
from django.conf import settings
from django.shortcuts import render, redirect
from climate_hazards_analysis.utils.climate_hazards_analysis import generate_climate_hazards_analysis
from climate_hazards_analysis.utils.generate_report import generate_climate_hazards_report_pdf

# Use a local UPLOAD_DIR variable defined in this views.py file
UPLOAD_DIR = os.path.join(settings.BASE_DIR, 'climate_hazards_analysis', 'static', 'input_files')

def process_data(data):
    """
    Replace NaN values in a list of dictionaries with custom strings.
    For 'Sea Level Rise', use "Little to no effect".
    For 'Tropical Cyclones', use "Data not available".
    Otherwise, use "N/A".
    """
    for row in data:
        for key, value in row.items():
            # print(f"key: {key}")
            if pd.isna(value):
                if key == 'Elevation (meter above sea level)' or key =='2030 Sea Level Rise (in meters)' or key =='2040 Sea Level Rise (in meters)' or key =='2050 Sea Level Rise (in meters)' or key =='2060 Sea Level Rise (in meters)':
                    row[key] = "Little to no effect"
                elif key == '1-min Maximum Sustain Windspeed 10 year Return Period (km/h)' or key == '1-min Maximum Sustain Windspeed 20 year Return Period (km/h)' or key == '1-min Maximum Sustain Windspeed 50 year Return Period (km/h)' or key == '1-min Maximum Sustain Windspeed 100 year Return Period (km/h)':
                    row[key] = "Data not available"
                else:
                    row[key] = "N/A"
    return data

def upload_facility_csv(request):
    """
    Handles the upload of facility CSV files and stores selected climate hazards.
    """
    # Define the list of available climate hazard fields for the checkboxes
    climate_hazards_fields = [
        'Flood',
        'Water Stress',
        'Sea Level Rise', 
        'Tropical Cyclones',
        'Heat',
        'Storm Surge',
        'Rainfall Induced Landslide'
    ]
    
    if request.method == 'POST' and request.FILES.get('facility_csv'):
        os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure directory exists

        file = request.FILES['facility_csv']
        file_path = os.path.join(UPLOAD_DIR, file.name)

        # Save the uploaded file
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        # Store file path in session
        request.session['facility_csv_path'] = file_path

        # Retrieve the list of selected climate hazards from the checkboxes
        selected_fields = request.POST.getlist('fields')
        request.session['selected_dynamic_fields'] = selected_fields

        print("Uploaded facility CSV file path:", file_path)
        print("Selected climate hazards:", selected_fields)

        return redirect('climate_hazards_analysis:climate_hazards_analysis')
    
    # For GET requests, render the upload form with the climate hazard checkboxes
    # Use just the template name as it should be within the app's template directory
    return render(request, 'upload.html', {
        'climate_hazards_fields': climate_hazards_fields
    })

def climate_hazards_analysis(request):
    """
    Processes the uploaded facility CSV and selected climate hazards.
    Generates a combined analysis and displays the results.
    """
    # Define the list of available climate hazard fields for the checkboxes
    climate_hazards_fields = [
        'Flood',
        'Water Stress',
        'Sea Level Rise',
        'Tropical Cyclones',
        'Heat',
        'Storm Surge',
        'Rainfall Induced Landslide'
    ]
    
    # Retrieve the uploaded facility CSV file path from the session.
    facility_csv_path = request.session.get('facility_csv_path')
    if not facility_csv_path or not os.path.exists(facility_csv_path):
        return render(request, 'upload.html', {
            'error': 'No facility file uploaded or file not found.',
            'climate_hazards_fields': climate_hazards_fields  # Include fields in error response
        })

    # Retrieve the list of selected climate hazards from the session.
    selected_fields = request.session.get('selected_dynamic_fields', [])
    print("Climate Hazards selected:", selected_fields)

    # Call the climate hazards analysis function with only the required parameters
    result = generate_climate_hazards_analysis(
        facility_csv_path=facility_csv_path,
        selected_fields=selected_fields
    )

    # Check for errors in the result.
    if result is None or 'error' in result:
        error_message = result.get('error', 'Unknown error') if result else 'Analysis failed.'
        # Use existing error template if available, or fall back to upload template
        try:
            return render(request, 'error.html', {
                'error': error_message,
                'climate_hazards_fields': climate_hazards_fields  # Include fields in error response
            })
        except:
            return render(request, 'upload.html', {
                'error': error_message,
                'climate_hazards_fields': climate_hazards_fields  # Include fields in error response
            })

    # Get the path to the combined output CSV.
    combined_csv_path = result.get('combined_csv_path')
    if not combined_csv_path or not os.path.exists(combined_csv_path):
        return render(request, 'upload.html', {
            'error': 'Combined CSV output not found.',
            'climate_hazards_fields': climate_hazards_fields  # Include fields in error response
        })

    # Load the combined CSV file.
    df = pd.read_csv(combined_csv_path)
    data = df.to_dict(orient="records")
    columns = df.columns.tolist()

    # Get the paths to any generated plots.
    plot_path = result.get('plot_path')
    all_plots = result.get('all_plots', [])

    context = {
        'data': data,
        'columns': columns,
        'plot_path': plot_path,
        'all_plots': all_plots,
        'selected_dynamic_fields': selected_fields,
        'climate_hazards_fields': climate_hazards_fields  # Add climate_hazards_fields to the context
    }

    # Use the existing analysis template if available
    template_name = 'climate_hazards_analysis.html'
    
    # Fallback options if needed
    try:
        return render(request, template_name, context)
    except:
        # If the template doesn't exist, try alternative template names
        alternative_templates = [
            'climate_hazard_analysis.html',
            'climate_hazards.html',
            'analysis.html'
        ]
        
        for alt_template in alternative_templates:
            try:
                return render(request, alt_template, context)
            except:
                continue
        
        # If all else fails, use the upload template to display a basic result
        return render(request, 'upload.html', {
            'data': data,
            'columns': columns,
            'error': 'Analysis completed but display template not found.',
            'climate_hazards_fields': climate_hazards_fields  # Include fields in basic result
        })

def water_stress_mapbox_fetch(request):
    return render(request, 'water_stress_mapbox.html')

def flood_exposure_mapbox_fetch(request):
    return render(request, 'flood_exposure_mapbox.html')

def heat_exposure_mapbox_fetch(request):
    return render(request, 'heat_exposure_mapbox.html')

def sea_level_rise_mapbox_fetch(request):
    return render(request, 'sea_level_rise_mapbox.html')

def tropical_cyclone_mapbox_fetch(request):
    return render(request, 'tropical_cyclone_mapbox.html')

def multi_hazard_mapbox_fetch(request):
    selected_dynamic_fields = request.session.get('selected_dynamic_fields', [])
    return render(request, 'multi_hazard_mapbox.html', {'selected_dynamic_fields': selected_dynamic_fields})

def generate_report(request):
    """
    Django view that generates the PDF report and returns it as an HTTP response.
    """
    selected_fields = request.session.get('selected_dynamic_fields', [])
    buffer = BytesIO()
    generate_climate_hazards_report_pdf(buffer, selected_fields)  # Use the updated function name
    pdf = buffer.getvalue()
    buffer.close()
    response = HttpResponse(pdf, content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="Climate_Hazard_Exposure_Report.pdf"'
    return response