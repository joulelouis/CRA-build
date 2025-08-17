from django.urls import path

from .views import OverrideRequestCreateView, OverrideRequestListView

urlpatterns = [
    path('overrides/', OverrideRequestCreateView.as_view(), name='override_request'),
    path('overrides/all/', OverrideRequestListView.as_view(), name='override_request_list'),
]