from django.urls import path
from .views import tc_upload_facility_csv

app_name = "tropical_cyclone_analysis"

urlpatterns = [
    path('', tc_upload_facility_csv, name='tc_upload_facility_csv'),
]