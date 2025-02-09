import os
import pandas as pd
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render, redirect
from .utils.water_stress import generate_water_stress_plot  # Ensure you import the function

UPLOAD_DIR = os.path.join(settings.BASE_DIR, 'water_stress/static/input_files')


def upload_water_stress_facility_csv(request):
    # List of available fields for the water stress analysis
    available_fields = [
        'bws_05_cat',
        'bws_05_lab',
        'iav_06_cat',
        'iav_06_lab',
        'bws_06_cat',
        'bws_06_lab'
    ]
    
    if request.method == 'POST' and request.FILES.get('facility_csv'):
        os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure the upload directory exists

        file = request.FILES['facility_csv']
        file_path = os.path.join(UPLOAD_DIR, file.name)

        # Save the uploaded file
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        # Store file path in session
        request.session['water_stress_facility_csv_path'] = file_path

        # Retrieve the list of selected dynamic fields from the checkboxes
        selected_fields = request.POST.getlist('fields')
        request.session['selected_dynamic_fields'] = selected_fields

        print("Uploaded facility CSV file path:", file_path)
        print("Selected dynamic fields:", selected_fields)

        return redirect('water_stress:water_stress')  # Redirect to table & plot page

    # For GET requests, render the upload form with the dynamic checkboxes.
    context = {
        'available_fields': available_fields,
    }
    return render(request, 'water_stress/upload.html', context)

def water_stress(request):
    """
    Processes uploaded CSV, generates a table & plot, and displays them.
    Also includes the available_fields in the context.
    """
    # Define the full list of available fields
    available_fields = [
        'bws_05_cat',
        'bws_05_lab',
        'iav_06_cat',
        'iav_06_lab',
        'bws_06_cat',
        'bws_06_lab'
    ]
    
    file_path = request.session.get('water_stress_facility_csv_path')
    if not file_path or not os.path.exists(file_path):
        return render(request, 'water_stress/upload.html', {'error': 'No file uploaded or file not found.'})

    shapefile_base = os.path.join(UPLOAD_DIR, 'hybas_lake_au_lev06_v1c')
    required_files = [".shp", ".dbf", ".shx"]
    missing_files = [ext for ext in required_files if not os.path.exists(f"{shapefile_base}{ext}")]
    if missing_files:
        return HttpResponse(f"Missing shapefile components: {', '.join(missing_files)}", status=500)

    shapefile_path = f"{shapefile_base}.shp"
    dbf_path = f"{shapefile_base}.dbf"
    shx_path = f"{shapefile_base}.shx"
    csv_path = os.path.join(UPLOAD_DIR, 'Aqueduct40_baseline_monthly_y2023m07d05.csv')

    # Retrieve the list of fields selected by the user
    selected_fields = request.session.get('selected_dynamic_fields', None)
    print("Using selected fields for plotting:", selected_fields)

    # Pass the selected fields as dynamic_fields and plot_fields.
    # (If no fields were selected, you might want to decide on a default.)
    plot_path = generate_water_stress_plot(
        shapefile_path,
        dbf_path,
        shx_path,
        csv_path,
        file_path,
        dynamic_fields=selected_fields,
        plot_fields=selected_fields
    )

    # Load updated CSV with facility locations
    updated_csv_path = os.path.join(UPLOAD_DIR, 'sample_locs.csv')
    if os.path.exists(updated_csv_path):
        df = pd.read_csv(updated_csv_path)
        data = df.to_dict(orient="records")
        columns = df.columns.tolist()
    else:
        data, columns = [], []

    # Include the available_fields in the context
    return render(request, 'water_stress/water_stress.html', {
        'data': data,
        'columns': columns,
        'plot_path': plot_path,
        'available_fields': available_fields
    })

def water_stress_image(request):
    """
    Returns the generated water stress plot as an image.
    """
    plot_path = os.path.join(UPLOAD_DIR, 'water_stress_plot.png')

    if os.path.exists(plot_path):
        with open(plot_path, "rb") as image_file:
            return HttpResponse(image_file.read(), content_type="image/png")
    else:
        return HttpResponse("❌ Error: Water Stress Plot not found!", status=404)
