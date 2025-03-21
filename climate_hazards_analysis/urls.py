from django.urls import path
from .views import (
    upload_facility_csv,
    climate_hazards_analysis,
    water_stress_mapbox_ajax,
    flood_exposure_mapbox_ajax,
    heat_exposure_mapbox_ajax,
    sea_level_rise_mapbox_ajax,
    tropical_cyclone_mapbox_ajax,
    generate_report,  # newly added
)

app_name = "climate_hazards_analysis"

urlpatterns = [
    path('', upload_facility_csv, name='upload_facility_csv'),
    path('output-with-exposure/', climate_hazards_analysis, name='climate_hazards_analysis'),
    path('water-stress-mapbox/', water_stress_mapbox_ajax, name='water_stress_mapbox_ajax'),
    path('flood-exposure-mapbox/', flood_exposure_mapbox_ajax, name='flood_exposure_mapbox_ajax'),
    path('heat-exposure-mapbox/', heat_exposure_mapbox_ajax, name='heat_exposure_mapbox_ajax'),
    path('sea-level-rise-mapbox/', sea_level_rise_mapbox_ajax, name='sea_level_rise_mapbox_ajax'),
    path('tropical-cyclone-mapbox/', tropical_cyclone_mapbox_ajax, name='tropical_cyclone_mapbox_ajax'),
    path('generate-report/', generate_report, name='generate_report'),
]