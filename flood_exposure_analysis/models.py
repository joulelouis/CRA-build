import geocoder
from django.db import models

mapbox_access_token = 'pk.eyJ1Ijoiam91bGUyMzMxIiwiYSI6ImNtNzMyczF5YTBmdHIybHB3bXVqMWdiaGgifQ.WW8zxem2Mfu8jPk-L_kSdA'

# Create your models here.
class Address(models.Model):
    address = models.TextField()
    lat = models.FloatField(blank=True, null=True)
    long = models.FloatField(blank=True, null=True)

    def save(self, *args, **kwargs):
        if self.address:
            g = geocoder.mapbox(self.address, key=mapbox_access_token)
            
            if g.ok:
                latlng = g.latlng #returns [lat, long]
                self.lat = latlng[0]
                self.long = latlng[1]
            else:
                self.lat = None
                self.long = None

        return super(Address, self).save(*args, **kwargs)