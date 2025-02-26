import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render, redirect
from climate_hazards_analysis.utils.climate_hazards_analysis import generate_climate_hazards_analysis

# Use a local UPLOAD_DIR variable defined in this views.py file
UPLOAD_DIR = os.path.join(settings.BASE_DIR, 'climate_hazards_analysis', 'static', 'input_files')

def upload_facility_csv(request):
    # List of climate hazards fields 
    climate_hazards_fields = [
        'Heat Exposure Analysis',
        'Soil Level Risk Exposure Analysis',
        'Flood Exposure Analysis',
        'Water Stress Analysis',
        'Tropical Cyclones',
        'Plot Hazard Maps',
    ]

    if request.method == 'POST' and request.FILES.get('facility_csv'):
        os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure directory exists

        file = request.FILES['facility_csv']
        file_path = os.path.join(UPLOAD_DIR, file.name)

        # Save the uploaded file
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        # Store file path in session using a consistent key
        request.session['climate_hazards_analysis_csv_path'] = file_path

        print("Uploaded facility CSV file path:", file_path)

        # Retrieve the list of selected climate hazards from the checkboxes
        selected_fields = request.POST.getlist('fields')
        request.session['selected_dynamic_fields'] = selected_fields

        print("Selected climate hazards:", selected_fields)

        return redirect('climate_hazards_analysis:climate_hazards_analysis')
    
    # For GET requests, render the upload form with the dynamic checkboxes.
    context = {
        'climate_hazards_fields': climate_hazards_fields,
    }
    return render(request, 'upload.html', context)


def climate_hazards_analysis(request):

    climate_hazards_fields = [
        'Heat Exposure Analysis',
        'Soil Level Risk Exposure Analysis',
        'Flood Exposure Analysis',
        'Water Stress Analysis',
        'Tropical Cyclones',
        'Plot Hazard Maps',
    ]




    # Define the required file paths using the local UPLOAD_DIR variable
    shapefile_base = os.path.join(UPLOAD_DIR, 'hybas_lake_au_lev06_v1c')
    shapefile_path = f"{shapefile_base}.shp"
    dbf_path = f"{shapefile_base}.dbf"
    shx_path = f"{shapefile_base}.shx"
    
    water_risk_csv_path = os.path.join(UPLOAD_DIR, 'Aqueduct40_baseline_monthly_y2023m07d05.csv')
    # Use the same session key as in the upload view
    facility_csv_path = request.session.get('climate_hazards_analysis_csv_path')
    raster_path = os.path.join(UPLOAD_DIR, 'Abra_Flood_100year.tif')
    
    # Check if facility CSV exists
    if not facility_csv_path or not os.path.exists(facility_csv_path):
        return render(request, 'climate_hazards_analysis/upload.html', {
            'error': 'No facility file uploaded or file not found.'
        })
    
    # Retrieve the list of climate hazards selected by the user
    selected_fields = request.session.get('selected_dynamic_fields', None)
    print("Climate Hazards selected:", selected_fields)
    
    # Call the combined analysis function
    result = generate_climate_hazards_analysis(
        shapefile_path, dbf_path, shx_path,
        water_risk_csv_path, facility_csv_path, raster_path, selected_fields
    )
    
    if result is None:
        return render(request, 'climate_hazards_analysis/error.html', {
            'error': 'Combined analysis failed. Please check logs for details.'
        })
    
    # Load the combined CSV into a DataFrame
    combined_csv_path = result.get('combined_csv_path')
    plot_path = result.get('plot_path')
    
    if os.path.exists(combined_csv_path):
        df = pd.read_csv(combined_csv_path)
        # Rename columns according to your requirements
        df.rename(columns={
            'Site': 'Facility',
            'Lat': 'Latitude',
            'Long': 'Longitude',
            'bws_06_lab': 'Water Stress Exposure',
            'Exposure': 'Flood Exposure'
        }, inplace=True)
        data = df.to_dict(orient="records")
        columns = df.columns.tolist()
    else:
        data, columns = [], []
    
    context = {
        'data': data,
        'columns': columns,
        'plot_path': plot_path,
        'climate_hazards_fields': climate_hazards_fields
    }
    
    return render(request, 'climate_hazards_analysis.html', context)
