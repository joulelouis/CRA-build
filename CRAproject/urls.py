"""
URL configuration for CRAproject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static
from accounts.views import login_page, signup_page

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.basepage), 
    # path('', views.homepage), 
    path('feature_1/', views.feature_1),
    path('feature_2/', views.feature_2),
    path('posts/', include('posts.urls')), #include the urls in the posts app in the urls in myproject (main project)
    path('delta_method/', views.delta_method),
    path('water-stress/', include('water_stress.urls')),
    path('flood-exposure-analysis/', include('flood_exposure_analysis.urls')),
    path('climate-hazards-analysis/', include('climate_hazards_analysis.urls')),
    path('sea-level-rise-analysis/', include('sea_level_rise_analysis.urls')),
    path('tropical-cyclone-analysis/', include('tropical_cyclone_analysis.urls')),
    path('climate-hazards-analysis-v2/', include('climate_hazards_analysis_v2.urls')),
    path('login/', login_page, name='login_page'),
    path('signup/', signup_page, name='signup_page'),
    path('api/auth/', include('accounts.urls')),
    path('api/', include('overrides.urls')),
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
