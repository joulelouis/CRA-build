from django.urls import path
from .views import tc_upload_facility_csv, tropical_cyclone_analysis

app_name = "tropical_cyclone_analysis"

urlpatterns = [
    path('', tc_upload_facility_csv, name='tc_upload_facility_csv'),
    path('output-tc', tropical_cyclone_analysis, name='tropical_cyclone_analysis'),
]