from django.shortcuts import render

def view_map(request):
    
    return render(request, 'climate_hazards_analysis_v2/climate_hazard_map.html')
