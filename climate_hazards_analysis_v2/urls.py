from django.urls import path
from .views import view_map

app_name = "climate_hazards_analysis_v2"

urlpatterns = [
    path('', view_map, name='view_map'),
]