from rest_framework import generics, permissions

from .models import OverrideRequest
from .serializers import OverrideRequestSerializer


class OverrideRequestCreateView(generics.CreateAPIView):
    serializer_class = OverrideRequestSerializer
    permission_classes = [permissions.IsAuthenticated]

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class OverrideRequestListView(generics.ListAPIView):
    serializer_class = OverrideRequestSerializer
    permission_classes = [permissions.IsAdminUser]
    queryset = OverrideRequest.objects.all()