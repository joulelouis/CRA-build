from django.urls import path
from .views import upload_facility_csv, climate_hazards_analysis

app_name = "climate_hazards_analysis"

urlpatterns = [
    path('', upload_facility_csv, name='upload_facility_csv'),
    path('output-with-exposure/', climate_hazards_analysis, name='climate_hazards_analysis'),
    # path('flood-exposure-map/', AddressView.as_view(), name='flood_exposure_map'),
]
