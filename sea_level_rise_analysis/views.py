import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render, redirect


# Use a local UPLOAD_DIR variable defined in this views.py file
UPLOAD_DIR = os.path.join(settings.BASE_DIR, 'sea_level_rise_analysis', 'static', 'input_files')

def upload_facility_csv(request):
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
    return render(request, 'upload.html', context)
