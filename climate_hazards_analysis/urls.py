from django.urls import path
from .views import (
    upload_facility_csv,
    climate_hazards_analysis,
    water_stress_mapbox_fetch,
    flood_exposure_mapbox_fetch,
    heat_exposure_mapbox_fetch,
    sea_level_rise_mapbox_fetch,
    tropical_cyclone_mapbox_fetch,
    multi_hazard_mapbox_fetch,
    generate_report,  
)

app_name = "climate_hazards_analysis"

urlpatterns = [
    path('', upload_facility_csv, name='upload_facility_csv'),
    path('output-with-exposure/', climate_hazards_analysis, name='climate_hazards_analysis'),
    path('water-stress-mapbox/', water_stress_mapbox_fetch, name='water_stress_mapbox_fetch'),
    path('flood-exposure-mapbox/', flood_exposure_mapbox_fetch, name='flood_exposure_mapbox_fetch'),
    path('heat-exposure-mapbox/', heat_exposure_mapbox_fetch, name='heat_exposure_mapbox_fetch'),
    path('sea-level-rise-mapbox/', sea_level_rise_mapbox_fetch, name='sea_level_rise_mapbox_fetch'),
    path('tropical-cyclone-mapbox/', tropical_cyclone_mapbox_fetch, name='tropical_cyclone_mapbox_fetch'),
    path('multi-hazard-mapbox/', multi_hazard_mapbox_fetch, name='multi_hazard_mapbox_fetch'),
    path('generate-report/', generate_report, name='generate_report'),
]