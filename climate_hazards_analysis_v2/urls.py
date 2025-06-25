from django.urls import path
from .views import view_map, get_facility_data, add_facility, select_hazards, show_results, generate_report, sensitivity_parameters, sensitivity_results, save_table_changes, reset_table_data, preview_uploaded_file

app_name = "climate_hazards_analysis_v2"

urlpatterns = [
    path('', view_map, name='view_map'),
    path('select-hazards/', select_hazards, name='select_hazards'),
    path('results/', show_results, name='show_results'),
    path('sensitivity-parameters/', sensitivity_parameters, name='sensitivity_parameters'),
    path('sensitivity-results/', sensitivity_results, name='sensitivity_results'),
    path('generate-report/', generate_report, name='generate_report'),
    path('api/facility-data/', get_facility_data, name='get_facility_data'),
    path('api/add-facility/', add_facility, name='add_facility'),
    path('api/preview-upload/', preview_uploaded_file, name='preview_uploaded_file'),
    path('save-table-changes/', save_table_changes, name='save_table_changes'),
    path('reset-table-data/', reset_table_data, name='reset_table_data'),
]