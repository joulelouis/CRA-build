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
    Generate a PDF report of climate hazard exposure analysis with each hazard on a separate page.
    
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
    
    # ---- FIRST PAGE ELEMENTS ----
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

    section_title_style = ParagraphStyle(
        'sectionTitle',
        parent=styles["Normal"],
        alignment=TA_LEFT,         # left alignment
        fontName="Helvetica-Bold", # bold font
        fontSize=14,               # larger font size
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
            Paragraph("Days over 30°C: <strong><font color='red'>High</font></strong>, <br/>Days over 33°C: <strong><font color='red'>High</font></strong>, <br/>Days over 35°C: <strong><font color='orange'>Medium</font></strong>", wrap_style),
            Paragraph("Lorem Ipsum Dolor with a very long explanation for the Heat Exposure Analysis", wrap_style)
        ])

    if "Flood" in selected_fields:
        overview_data.append([
            Paragraph("Flood", wrap_style),
            Paragraph("<strong><font color='green'>Low</font></strong>", wrap_style),
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
            Paragraph("1-min Maximum Sustain Windspeed 10 yr RP: <strong><font color='orange'>Medium</font></strong>, <br/>1-min Maximum Sustain Windspeed 20 yr RP: <strong><font color='orange'>Medium</font></strong>, <br/>1-min Maximum Sustain Windspeed 50 yr RP: <strong><font color='orange'>Medium</font></strong>, <br/>1-min Maximum Sustain Windspeed 100 yr RP: <strong><font color='orange'>Medium</font></strong>", wrap_style),
            Paragraph("Lorem Ipsum Dolor with a very long explanation for the Tropical Cyclone Exposure Analysis", wrap_style)
        ])

    if "Storm Surge" in selected_fields:
        overview_data.append([
            Paragraph("Storm Surge", wrap_style),
            Paragraph("Low, Medium, or High", wrap_style),
            Paragraph("Lorem Ipsum Dolor with a very long explanation for the Storm Surge", wrap_style)
        ])

    if "Rainfall Induced Landslide" in selected_fields:
        overview_data.append([
            Paragraph("Rainfall Induced Landslide", wrap_style),
            Paragraph("Low, Medium, High, or Generally Stable", wrap_style),
            Paragraph("Lorem Ipsum Dolor with a very long explanation for the Rainfall Induced Landslide", wrap_style)
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
    
    # ----- Helper function to safely create resized images -----
    def create_safe_image(image_path, max_width, max_height):
        try:
            if os.path.exists(image_path):
                img = Image(image_path)
                
                # Get image dimensions and calculate aspect ratio
                from PIL import Image as PILImage
                pil_img = PILImage.open(image_path)
                img_width, img_height = pil_img.size
                aspect_ratio = img_width / img_height
                
                # Calculate new dimensions to fit within constraints
                if img_width > max_width:
                    new_width = max_width
                    new_height = new_width / aspect_ratio
                else:
                    new_width = img_width
                    new_height = img_height
                
                # Further adjust if height exceeds maximum
                if new_height > max_height:
                    new_height = max_height
                    new_width = new_height * aspect_ratio
                
                # Set image dimensions
                img.drawWidth = new_width
                img.drawHeight = new_height
                
                return img
            else:
                # Return placeholder text if image not found
                return Paragraph(f"Image not found: {os.path.basename(image_path)}", styles["Normal"])
        except Exception as e:
            # Return error message if something goes wrong
            return Paragraph(f"Error loading image: {str(e)}", styles["Normal"])
    
    # ----- Helper function to create a hazard section -----
    def create_hazard_section(hazard_name, map_paths, assets_list):
        """Creates a hazard section with maps on the left and assets on the right."""
        section_elements = []
        
        try:
            # Calculate dimensions
            left_column_width = doc.width * 0.55  # 55% for maps
            right_column_width = doc.width * 0.35  # 35% for asset list
            
            # Calculate available height for maps based on number of maps
            available_height_per_map = (doc.height - 200) / len(map_paths)
            
            # Create maps column elements
            maps_col = []
            for i, map_path in enumerate(map_paths):
                # Always use "Asset Map" as the title
                map_title = Paragraph("Asset Map", styles["Heading4"])
                maps_col.append(map_title)
                maps_col.append(Spacer(1, 5))
                
                # Create and add map image
                map_img = create_safe_image(map_path, left_column_width * 0.9, available_height_per_map - 20)
                maps_col.append(map_img)
                
                # Add spacing between maps if not the last one
                if i < len(map_paths) - 1:
                    maps_col.append(Spacer(1, 15))
            
            # Create assets column elements
            assets_col = [
                Paragraph("Assets", styles["Heading4"]),
                Spacer(1, 5)
            ]
            
            # Add each asset to the list with spacing
            for asset in assets_list:
                assets_col.append(Paragraph(asset, styles["Normal"]))
                assets_col.append(Spacer(1, 5))
            
            # Create individual tables for each column
            maps_table = Table([[item] for item in maps_col], colWidths=[left_column_width * 0.9])
            maps_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            
            assets_table = Table([[item] for item in assets_col], colWidths=[right_column_width * 0.9])
            assets_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            
            # Create the main two-column layout
            layout_data = [[maps_table, assets_table]]
            layout_table = Table(layout_data, colWidths=[left_column_width, right_column_width])
            layout_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ]))
            
            # Add the layout to the section elements
            section_elements.append(layout_table)
            
        except Exception as e:
            # If anything goes wrong with layout creation, add error message instead
            error_msg = Paragraph(f"Error creating {hazard_name} section: {str(e)}", styles["Normal"])
            section_elements.append(error_msg)
        
        return section_elements
    
    # ----- Helper function to create a page for a hazard -----
    def add_hazard_page(hazard_name, map_paths, assets_list):
        """Creates a complete page for a hazard with header, title, and content."""
        # Start with a page break
        elements.append(PageBreak())
        
        # Add the header
        elements.append(header_table)
        elements.append(Spacer(1, 12))
        
        # Add "Assets with High Hazard Rating" title
        high_hazard_title = Paragraph("Assets with High Hazard Rating", table_title_style)
        elements.append(high_hazard_title)
        elements.append(Spacer(1, 20))
        
        # Add hazard section title
        hazard_title = Paragraph(hazard_name, section_title_style)
        elements.append(hazard_title)
        elements.append(Spacer(1, 15))
        
        # Add the hazard content
        hazard_content = create_hazard_section(hazard_name, map_paths, assets_list)
        elements.extend(hazard_content)
    
    # Define map paths
    luzon_map_path = os.path.join(settings.BASE_DIR, "climate_hazards_analysis", "static", "images", "luzon.png")
    mindanao_map_path = os.path.join(settings.BASE_DIR, "climate_hazards_analysis", "static", "images", "mindanao.png")
    
    # Add a page for Heat if selected
    if "Heat" in selected_fields:
        # Heat assets
        heat_assets = [
            "SM Manila Shakeys Branch",
            "Commissary in Paranaque",
            "Coconut Processing Plant in Gen San",
            "Tuna Processing Plant in Gen San"
        ]
        
        # Add Heat page
        add_hazard_page(
            "Heat",
            [luzon_map_path, mindanao_map_path], # Both Luzon and Mindanao maps
            heat_assets
        )
    
    # Add a page for Water Stress if selected
    if "Water Stress" in selected_fields:
        # Water Stress assets
        water_stress_assets = [
            "SM Manila Shakeys Branch",
            "Commissary in Paranaque"
        ]
        
        # Add Water Stress page
        add_hazard_page(
            "Water Stress",
            [luzon_map_path],  # Only Luzon map
            water_stress_assets
        )
    
    # Add a page for Storm Surge if selected
    if "Storm Surge" in selected_fields:
        # Storm Surge assets
        storm_surge_assets = [
            "SM Manila Shakeys Branch"
        ]
        
        # Add Storm Surge page
        add_hazard_page(
            "Storm Surge",
            [luzon_map_path],  # Only Luzon map
            storm_surge_assets
        )
    
    # Build the document
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