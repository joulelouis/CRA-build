from django.contrib import admin
from django.utils import timezone

from .models import OverrideRequest


@admin.register(OverrideRequest)
class OverrideRequestAdmin(admin.ModelAdmin):
    list_display = ('user', 'is_approved', 'approved_by', 'created_at')
    actions = ['approve_requests']

    def approve_requests(self, request, queryset):
        for obj in queryset.filter(is_approved=False):
            obj.approve(request.user)
    approve_requests.short_description = 'Approve selected override requests'