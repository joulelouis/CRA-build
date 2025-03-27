import os
import pandas as pd
from io import BytesIO
from django.http import HttpResponse
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, Image
from reportlab.lib import colors
import datetime
from django.conf import settings
from django.shortcuts import render, redirect
from climate_hazards_analysis.utils.climate_hazards_analysis import generate_climate_hazards_analysis

# Use a local UPLOAD_DIR variable defined in this views.py file
UPLOAD_DIR = os.path.join(settings.BASE_DIR, 'climate_hazards_analysis', 'static', 'input_files')

def process_data(data):
    """
    Replace NaN values in a list of dictionaries with "N/A".
    """
    for row in data:
        for key, value in row.items():
            if pd.isna(value):
                row[key] = "N/A"
    return data

def upload_facility_csv(request):
    # List of climate hazards fields 
    climate_hazards_fields = [
        'Heat',
        'Sea Level Rise',
        'Flood',
        'Water Stress',
        'Tropical Cyclones',
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
        'Heat',
        'Sea Level Rise',
        'Flood',
        'Water Stress',
        'Tropical Cyclones',
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
            'n>30degC_2125': 'Days over 30 degrees Celsius',
            'n>33degC_2125': 'Days over 33 degrees Celsius',
            'n>35degC_2125': 'Days over 35 degrees Celsius',
            'bws_06_raw': 'Water Stress Exposure (in %)',
            'Exposure': 'Flood Depth',
            'SRTM elevation': 'Elevation (meter above sea level)',
            '2030 Sea Level Rise Cl 0.5': '2030 Sea Level Rise (in meters)',
            '2040 Sea Level Rise Cl 0.5': '2040 Sea Level Rise (in meters)',
            '2050 Sea Level Rise Cl 0.5': '2050 Sea Level Rise (in meters)',
            '2060 Sea Level Rise Cl 0.5': '2060 Sea Level Rise (in meters)',
            '1-min MSW 10 yr RP': '1-min Maximum Sustain Windspeed 10 year Return Period (km/h)',
            '1-min MSW 20 yr RP': '1-min Maximum Sustain Windspeed 20 year Return Period (km/h)',
            '1-min MSW 50 yr RP': '1-min Maximum Sustain Windspeed 50 year Return Period (km/h)',
            '1-min MSW 100 yr RP': '1-min Maximum Sustain Windspeed 100 year Return Period (km/h)',
        }, inplace=True)
        data = df.to_dict(orient="records")
        # Process data to replace any NaN values with "N/A"
        data = process_data(data)
        columns = df.columns.tolist()
    else:
        data, columns = [], []
    
    context = {
        'data': data,
        'columns': columns,
        'plot_path': plot_path,
        'climate_hazards_fields': climate_hazards_fields,
        'selected_dynamic_fields': request.session.get('selected_dynamic_fields', []),
    }

    return render(request, 'climate_hazards_analysis.html', context)

def water_stress_mapbox_ajax(request):
    return render(request, 'water_stress_mapbox.html')

def flood_exposure_mapbox_ajax(request):
    return render(request, 'flood_exposure_mapbox.html')

def heat_exposure_mapbox_ajax(request):
    return render(request, 'heat_exposure_mapbox.html')

def sea_level_rise_mapbox_ajax(request):
    return render(request, 'sea_level_rise_mapbox.html')

def tropical_cyclone_mapbox_ajax(request):
    return render(request, 'tropical_cyclone_mapbox.html')


def generate_building_report_pdf(buffer, selected_fields):
    """
    Generate a PDF report using ReportLab and write it into the provided buffer.
    """
    # Create the document template
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    # Define a style for the header title
    header_title_style = ParagraphStyle(
        "HeaderTitle",
        parent=styles["Heading1"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24
    )
    
    # Build the logo image from the specified path
    logo_path = os.path.join(settings.BASE_DIR, "climate_hazards_analysis", "static", "images", "sgv-logo.png")
    logo = Image(logo_path)
    logo.drawHeight = 50  # adjust as needed
    logo.drawWidth = 50   # adjust as needed

    # Wrap the logo in a table cell to add a black background
    logo_table = Table([[logo]], colWidths=[logo.drawWidth])
    logo_table.hAlign = 'LEFT'
    logo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOX', (0, 0), (-1, -1), 0, colors.white),
    ]))

    # Create the header title paragraph
    title_paragraph = Paragraph("Physical Climate Risk Report", header_title_style)
    
    # Calculate the column widths:
    # We reserve the left column for the logo (plus a small margin), and the right column the same width.
    left_col_width = logo.drawWidth + 10
    middle_col_width = doc.width - 2 * left_col_width

    # Create a three-column header table: [Logo] [Title] [Empty]
    header_data = [[logo_table, title_paragraph, ""]]
    header_table = Table(header_data, colWidths=[left_col_width, middle_col_width, left_col_width])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (0,0), 'LEFT'),
        ('ALIGN', (1,0), (1,0), 'CENTER'),
        ('ALIGN', (2,0), (2,0), 'LEFT'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    
    elements = []
    # Insert the header table at the very top
    elements.append(header_table)
    elements.append(Spacer(1, 12))
    
    # Download info
    download_time = datetime.datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
    download_info = Paragraph(f"Downloaded: {download_time}", styles["Normal"])
    elements.append(download_info)
    elements.append(Spacer(1, 24))

    table_title_style = ParagraphStyle(
        'tableTitle',
        parent=styles["Normal"],
        alignment=TA_CENTER,       # center alignment
        fontName="Helvetica-Bold", # bold font
        fontSize=12,               # larger font size
        leading=20                 # optional: adjust line spacing
    )

    # Project Overview Table
    table_title = Paragraph("Climate Hazards Assessment", table_title_style)
    elements.append(table_title)
    styles = getSampleStyleSheet()
    wrap_style = styles["Normal"]

    overview_data = [
        [Paragraph("Climate Hazard", wrap_style),
         Paragraph("Portfolio Exposure Rating", wrap_style),
         Paragraph("Explanation and Recommendation", wrap_style)]
    ]

    # For each hazard, check if it is in selected_fields before appending its row.
    if "Heat" in selected_fields:
        overview_data.append([
            Paragraph("Heat", wrap_style),
            Paragraph("Days over 30°C: <strong>No. of Days</strong>, <br/>Days over 33°C: <strong>No. of Days</strong>, <br/>Days over 35°C: <strong>No. of Days</strong>", wrap_style),
            Paragraph("Lorem Ipsum Dolor with a very long explanation for the Heat Exposure Analysis", wrap_style)
        ])

    if "Flood" in selected_fields:
        overview_data.append([
            Paragraph("Flood", wrap_style),
            Paragraph("Low, Medium, or High", wrap_style),
            Paragraph("Lorem Ipsum Dolor with a very long explanation for the Flood Exposure Analysis", wrap_style)
        ])

    if "Water Stress" in selected_fields:
        overview_data.append([
            Paragraph("Water Stress", wrap_style),
            Paragraph("Low, Medium, or High", wrap_style),
            Paragraph("Lorem Ipsum Dolor with a very long explanation for the Water Stress Exposure Analysis", wrap_style)
        ])

    if "Sea Level Rise" in selected_fields:
        overview_data.append([
            Paragraph("Sea Level Rise", wrap_style),
            Paragraph("Elevation (m above sea level): <strong>value</strong>, <br/>2030 Sea Level Rise (m): <strong>value</strong>, <br/>2040 Sea Level Rise (m): <strong>value</strong>, <br/>2050 Sea Level Rise (m): <strong>value</strong>, <br/>2060 Sea Level Rise (m): <strong>value</strong>", wrap_style),
            Paragraph("Lorem Ipsum Dolor with a very long explanation for the Sea Level Rise Exposure Analysis", wrap_style)
        ])

    if "Tropical Cyclones" in selected_fields:
        overview_data.append([
            Paragraph("Tropical Cyclone", wrap_style),
            Paragraph("1-min Maximum Sustain Windspeed 10 yr RP: <strong>value</strong>, <br/>1-min Maximum Sustain Windspeed 20 yr RP: <strong>value</strong>, <br/>1-min Maximum Sustain Windspeed 50 yr RP: <strong>value</strong>, <br/>1-min Maximum Sustain Windspeed 100 yr RP: <strong>value</strong>", wrap_style),
            Paragraph("Lorem Ipsum Dolor with a very long explanation for the Tropical Cyclone Exposure Analysis", wrap_style)
        ])

    # Create your overview table with your data and styling
    available_width = doc.width
    # Define relative column widths: column 3 will be wider (50% of available width)
    col_widths = [available_width * 0.25, available_width * 0.25, available_width * 0.5]
    overview_table = Table(overview_data, colWidths=col_widths)
    overview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        # You can also add cell padding if desired:
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))

    # Wrap the table in a container table to provide overall margins.
    container = Table([[overview_table]], colWidths=[doc.width])
    container.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 8),   # adjust margin (in points)
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        
    ]))

    # Append the container instead of the table directly.
    elements.append(container)
    elements.append(Spacer(1, 24))

    doc.build(elements)

def generate_report(request):
    """
    Django view that generates the PDF report and returns it as an HTTP response.
    """
    selected_fields = request.session.get('selected_dynamic_fields', [])
    buffer = BytesIO()
    generate_building_report_pdf(buffer, selected_fields)  # Pass the selected_fields here
    pdf = buffer.getvalue()
    buffer.close()
    response = HttpResponse(pdf, content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="Physical_Climate_Risk_Report.pdf"'
    return response