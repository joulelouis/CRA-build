from django.conf import settings
from django.db import models
from django.utils import timezone


class OverrideRequest(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    reason = models.TextField()
    is_approved = models.BooleanField(default=False)
    approved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        related_name='approved_overrides',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    approved_at = models.DateTimeField(null=True, blank=True)

    def approve(self, admin_user):
        self.is_approved = True
        self.approved_by = admin_user
        self.approved_at = timezone.now()
        self.save()

    def __str__(self):
        return f"OverrideRequest({self.user}, approved={self.is_approved})"