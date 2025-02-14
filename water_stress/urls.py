from django.urls import path
from .views import upload_water_stress_facility_csv, water_stress, water_stress_image, AddressView

app_name = "water_stress"  # Add app name

urlpatterns = [
    path('', upload_water_stress_facility_csv, name='upload_water_stress_facility_csv'),
    path('updated-facility-locations/', water_stress, name='water_stress'),
    path('image/', water_stress_image, name='water_stress_image'),
    path('water-stress-map/', AddressView.as_view(), name='water_stress_map')
]
