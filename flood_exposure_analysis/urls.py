from django.urls import path
from .views import upload_facility_csv, flood_exposure_analysis

app_name = "flood_exposure_analysis"

urlpatterns = [
    path('', upload_facility_csv, name='upload_facility_csv'),
    path('output-with-exposure/', flood_exposure_analysis, name='flood_exposure_analysis'),
]
