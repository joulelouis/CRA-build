from django.urls import path
from .views import view_map, get_facility_data, add_facility

app_name = "climate_hazards_analysis_v2"

urlpatterns = [
    path('', view_map, name='view_map'),
    path('api/facility-data/', get_facility_data, name='get_facility_data'),
    path('api/add-facility/', add_facility, name='add_facility'),
]