from django.urls import path
from .views import upload_facility_csv

app_name = "climate_hazards_analysis"

urlpatterns = [
    path('', upload_facility_csv, name='upload_facility_csv'),
]