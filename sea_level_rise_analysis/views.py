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
    import os
    import pandas as pd

    # Retrieve the uploaded facility CSV file path from the session.
    facility_csv_path = request.session.get('sea_level_rise_analysis_csv_path')
    if not facility_csv_path or not os.path.exists(facility_csv_path):
        return render(request, 'sea_level_rise_analysis/upload.html', {
            'error': 'No facility file uploaded or file not found.'
        })

    # Retrieve any selected fields from the session.
    selected_fields = request.session.get('selected_dynamic_fields', None)
    print("Climate Hazards selected:", selected_fields)

    # Call the analysis function.
    result = generate_sea_level_rise_analysis(facility_csv_path)

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
            columns = df.columns.tolist()
        else:
            data, columns = [], []
    else:
        data, columns = [], []

    context = {
        'data': data,
        'columns': columns,
        'png_paths': png_paths,  # You can use this in your template to display or link to the PNGs.
        'sea_level_rise_fields': [],  # Update as needed.
        'selected_dynamic_fields': request.session.get('selected_dynamic_fields', []),
    }

    return render(request, 'sea_level_rise_analysis/sea_level_rise_analysis.html', context)