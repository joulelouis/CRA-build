import os
import pandas as pd
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render, redirect
from .utils.flood_exposure_analysis import generate_flood_exposure_analysis

UPLOAD_DIR = os.path.join(settings.BASE_DIR, 'flood_exposure_analysis/static/input_files')

# Create your views here.
def upload_facility_csv(request):
    """
    Handles file uploads and redirects to /flood-exposure-analysis/output-with-exposure/
    """
    if request.method == 'POST' and request.FILES.get('facility_csv'):
        os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure directory exists

        file = request.FILES['facility_csv']
        file_path = os.path.join(UPLOAD_DIR, file.name)

        # Save the uploaded file
        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        # Store file path in session
        request.session['flood_exposure_facility_csv_path'] = file_path

        print(request.session.get('flood_exposure_facility_csv_path'))

        return redirect('flood_exposure_analysis:flood_exposure_analysis')  # Redirect to table & plot page

    return render(request, 'flood_exposure_analysis/upload.html')  # Upload form template

def flood_exposure_analysis(request):

    file_path = request.session.get('flood_exposure_facility_csv_path')

    if not file_path or not os.path.exists(file_path):
        return render(request, 'flood_exposure_analysis/upload.html', {'error': 'No file uploaded or file not found.'})
    
    raster_path = os.path.join(UPLOAD_DIR, 'Abra_Flood_100year.tif')

    output_path = generate_flood_exposure_analysis(file_path, raster_path)

    updated_csv_path = os.path.join(UPLOAD_DIR, 'output_with_exposure.csv')

    if os.path.exists(updated_csv_path):
        df = pd.read_csv(updated_csv_path)
        data = df.to_dict(orient="records")
        columns = df.columns.tolist()
    else:
        data, columns = [], []

    return render(request, 'flood_exposure_analysis/flood_exposure_analysis.html', {
        'data': data,
        'columns': columns,
        'output_path': output_path
    })
