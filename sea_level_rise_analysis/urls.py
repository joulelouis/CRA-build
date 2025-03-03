from django.urls import path
from .views import slr_upload_facility_csv, sea_level_rise_analysis

app_name = "sea_level_rise_analysis"

urlpatterns = [
    path('', slr_upload_facility_csv, name='slr_upload_facility_csv'),
    path('output-slr', sea_level_rise_analysis, name='sea_level_rise_analysis'),
]