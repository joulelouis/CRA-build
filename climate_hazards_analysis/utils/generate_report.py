"""
PDF Report Generation Module for Climate Hazards Analysis

This module contains functions for generating PDF reports based on climate hazard analysis results.
"""

import os
import datetime
from io import BytesIO
from django.conf import settings
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, Image
from reportlab.lib import colors


def generate_climate_hazards_report_pdf(buffer, selected_fields):
    """
    Generate a PDF report of climate hazard exposure analysis and write it into the provided buffer.
    
    Args:
        buffer (BytesIO): Buffer to write the PDF content into
        selected_fields (list): List of selected climate hazard fields
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
    title_paragraph = Paragraph("Climate Hazard Exposure Report", header_title_style)
    
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
    table_title = Paragraph("Climate Hazard Exposure Assessment", table_title_style)
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


def generate_report_response(selected_fields):
    """
    Generate a PDF report with climate hazard data.
    
    Args:
        selected_fields (list): List of selected climate hazard fields
        
    Returns:
        BytesIO: Buffer containing the generated PDF
    """
    buffer = BytesIO()
    generate_climate_hazards_report_pdf(buffer, selected_fields)
    buffer.seek(0)
    return buffer