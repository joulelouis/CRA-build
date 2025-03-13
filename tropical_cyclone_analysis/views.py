import os
from django.conf import settings
from django.shortcuts import render, redirect
from tropical_cyclone_analysis.utils.tropical_cyclone_analysis import generate_tropical_cyclone_analysis

# Use a local UPLOAD_DIR variable defined in this views.py file
UPLOAD_DIR = os.path.join(settings.BASE_DIR, 'tropical_cyclone_analysis', 'static', 'input_files')

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
