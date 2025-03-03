import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render, redirect
from sea_level_rise_analysis.utils.sea_level_rise_analysis import generate_sea_level_rise_analysis


# Use a local UPLOAD_DIR variable defined in this views.py file
UPLOAD_DIR = os.path.join(settings.BASE_DIR, 'sea_level_rise_analysis', 'static', 'input_files')

def slr_upload_facility_csv(request):
    # List of sea level rise fields 
    sea_level_rise_fields = []

    if request.method == 'POST' and request.FILES.get('facility_csv'):
        os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure directory exists

        file = request.FILES['facility_csv']
        file_path = os.path.join(UPLOAD_DIR, file.name)

        # Save the uploaded file
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        # Store file path in session using a consistent key
        request.session['sea_level_rise_analysis_csv_path'] = file_path

        print("Uploaded facility CSV file path:", file_path)

        # Retrieve the list of selected climate hazards from the checkboxes
        selected_fields = request.POST.getlist('fields')
        request.session['selected_dynamic_fields'] = selected_fields

        print("Selected sea level rise fields:", selected_fields)

        return redirect('sea_level_rise_analysis:sea_level_rise_analysis')
    
    # For GET requests, render the upload form with the dynamic checkboxes.
    context = {
        'sea_level_rise_fields': sea_level_rise_fields,
    }
    return render(request, 'sea_level_rise_analysis/upload.html', context)

def sea_level_rise_analysis(request):

    # List of sea level rise fields 
    sea_level_rise_fields = []

    # Define the required file paths using the local UPLOAD_DIR variable
    # shapefile_base = os.path.join(UPLOAD_DIR, 'hybas_lake_au_lev06_v1c')
    # shapefile_path = f"{shapefile_base}.shp"
    # dbf_path = f"{shapefile_base}.dbf"
    # shx_path = f"{shapefile_base}.shx"
    
    # water_risk_csv_path = os.path.join(UPLOAD_DIR, 'Aqueduct40_baseline_monthly_y2023m07d05.csv')
    # Use the same session key as in the upload view
    facility_csv_path = request.session.get('sea_level_rise_analysis_csv_path')
    # raster_path = os.path.join(UPLOAD_DIR, 'Abra_Flood_100year.tif')
    
    # Check if facility CSV exists
    if not facility_csv_path or not os.path.exists(facility_csv_path):
        return render(request, 'sea_level_rise_analysis/upload.html', {
            'error': 'No facility file uploaded or file not found.'
        })
    
    # Retrieve the list of climate hazards selected by the user
    selected_fields = request.session.get('selected_dynamic_fields', None)
    print("Climate Hazards selected:", selected_fields)
    
    # Call the combined analysis function
    result = generate_sea_level_rise_analysis(facility_csv_path)
    
    if result is None:
        return render(request, 'climate_hazards_analysis/error.html', {
            'error': 'Combined analysis failed. Please check logs for details.'
        })
    
    # Load the combined CSV into a DataFrame
    combined_csv_path = result.get('combined_csv_path')
    # plot_path = result.get('plot_path')
    
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
        # 'plot_path': plot_path,
        'sea_level_rise_fields': sea_level_rise_fields,
        'selected_dynamic_fields': request.session.get('selected_dynamic_fields', []), #context to be passed in the tab list condition
    }

    return render(request, 'sea_level_rise_analysis/sea_level_rise_analysis.html', context)