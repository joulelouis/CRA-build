import os
import pandas as pd
from django.shortcuts import render, redirect
from django.conf import settings
from tropical_cyclone_analysis.utils.tropical_cyclone_analysis import generate_tropical_cyclone_analysis

# Use a local UPLOAD_DIR variable defined in this views.py file
UPLOAD_DIR = os.path.join(settings.BASE_DIR, 'tropical_cyclone_analysis', 'static', 'input_files')

def process_data(data):
    """
    Replace NaN values in a list of dictionaries with empty strings.
    """
    for row in data:
        for key, value in row.items():
            if pd.isna(value):
                row[key] = "N/A"
    return data

def tc_upload_facility_csv(request):
    # List of sea level rise fields 
    tropical_cyclone_fields = []

    if request.method == 'POST' and request.FILES.get('facility_csv'):
        os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure directory exists

        file = request.FILES['facility_csv']
        file_path = os.path.join(UPLOAD_DIR, file.name)

        # Save the uploaded file
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        # Store file path in session using a consistent key
        request.session['tropical_cyclone_analysis_csv_path'] = file_path

        print("Uploaded facility CSV file path:", file_path)

        # Retrieve the list of selected climate hazards from the checkboxes
        selected_fields = request.POST.getlist('fields')
        request.session['selected_dynamic_fields'] = selected_fields

        print("Selected tropical cyclone fields:", selected_fields)

        return redirect('tropical_cyclone_analysis:tropical_cyclone_analysis')
    
    # For GET requests, render the upload form with the dynamic checkboxes.
    context = {
        'tropical_cyclone_fields': tropical_cyclone_fields,
    }
    return render(request, 'tropical_cyclone_analysis/upload.html', context)

def tropical_cyclone_analysis(request):
    # Retrieve the uploaded facility CSV file path from the session.
    facility_csv_path = request.session.get('tropical_cyclone_analysis_csv_path')
    if not facility_csv_path or not os.path.exists(facility_csv_path):
        return render(request, 'tropical_cyclone_analysis/upload.html', {
            'error': 'No facility file uploaded or file not found.'
        })

    # Retrieve any selected fields from the session.
    selected_fields = request.session.get('selected_dynamic_fields', None)
    print("Climate Hazards selected:", selected_fields)

    # Call the analysis function.
    result = generate_tropical_cyclone_analysis(facility_csv_path)

    # Check for errors in the result.
    if result is None or "error" in result:
        return render(request, 'climate_hazards_analysis/error.html', {
            'error': 'Combined analysis failed. Please check logs for details.'
        })

    # Get the list of generated CSV and PNG file paths.
    combined_csv_paths = result.get('combined_csv_paths', [])
    png_paths = result.get('png_paths', [])
    
    # If multiple CSVs were generated, pick the first one for display.
    if combined_csv_paths:
        combined_csv_path = combined_csv_paths[0]
        if os.path.exists(combined_csv_path):
            df = pd.read_csv(combined_csv_path)
            data = df.to_dict(orient="records")
            # Process the data to replace NaN values with empty strings.
            data = process_data(data)
            columns = df.columns.tolist()
        else:
            data, columns = [], []
    else:
        data, columns = [], []

    context = {
        'data': data,
        'columns': columns,
        'png_paths': png_paths,  # You can use this in your template to display or link to the PNGs.
        'tropical_cyclone_fields': [],  # Update as needed.
        'selected_dynamic_fields': request.session.get('selected_dynamic_fields', []),
    }

    return render(request, 'tropical_cyclone_analysis/tropical_cyclone_analysis.html', context)
