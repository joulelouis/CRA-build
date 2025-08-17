from rest_framework import serializers

from .models import OverrideRequest


class OverrideRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = OverrideRequest
        fields = (
            'id',
            'reason',
            'is_approved',
            'approved_by',
            'created_at',
            'approved_at',
        )
        read_only_fields = ('is_approved', 'approved_by', 'created_at', 'approved_at')