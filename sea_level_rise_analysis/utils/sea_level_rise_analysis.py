

# Plot and/or write a map for a specific component, scenario, and year
import numpy as np              # Numpy
from netCDF4 import Dataset     # This package reads netcdf files
import matplotlib.pyplot as plt # Matplotlib's pyplot used to make the plots
import os                       # Extracts directory names etc
#from scipy.interpolate import interp2d
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import sys
from geopandas.tools import sjoin
import rioxarray as rx
import cartopy.crs as ccrs
import glob
from django.conf import settings



def generate_sea_level_rise_analysis(facility_csv_path):

    file = "total_ssp585_medium_confidence_values.nc"

    # functions taken from internet for converting SLR dataset into lat lon dimensions
    def example_run(years):
        confidence = 'Medium'

        # if confidence is 'medium', choose scenario from ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
        # if confidence is 'low', choose scenario from ['ssp126', 'ssp245','ssp585']
        scenario='ssp585'

        # For process, choose from
        #for process in ['glaciers','GIS','AIS','landwaterstorage','oceandynamics','verticallandmotion','total']:
        #    write_scenario_map(confidence, year, scenario, process)

        process='total'

        for yr in years:
            write_scenario_map(confidence, yr, scenario, process)

    def plot_scenario_map(confidence,year,scenario,process):

        # lat,lon coords and land-sea mask
        fn = file
        fh = Dataset(fn, 'r')
        fh.set_auto_mask(False)
        lat = fh['lat'][1030:].reshape(181,360)[:,0]
        lon = fh['lon'][1030:].reshape(181,360)[0,:]
        lon[lon<0]+=360
        fh.close()

        mask = compute_sea_mask(lat, lon)


        fn = file
        fh = Dataset(fn, 'r')
        fh.set_auto_mask(False)

        # Time steps
        years = fh['years'][:14]
        year_idx = np.where(years == year)[0][0]

        # Find quantiles
        l_quant = np.where(fh['quantiles'][:] == 0.05)[0][0]
        m_quant = np.where(fh['quantiles'][:] == 0.50)[0][0]
        h_quant = np.where(fh['quantiles'][:] == 0.95)[0][0]
        quants = np.array([l_quant,m_quant,h_quant])

        fh[quants,year_idx,1030:].reshape(3,181,360)/1000
        fh.close()

        gridlims = 0.9*np.max(grid)

        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        tst = axs[0].pcolormesh(lon,lat,mask*grid[0,:,:],shading='auto',vmin=-gridlims,vmax=gridlims,cmap="RdYlBu_r")
        axs[0].set_title("Lower bound")
        axs[0].contour(lon,lat,mask,[0.5],colors='k',linewidths=0.5)
        axs[1].pcolormesh(lon,lat,mask*grid[1,:,:],shading='auto',vmin=-gridlims,vmax=gridlims,cmap="RdYlBu_r")
        axs[1].contour(lon,lat,mask,[0.5],colors='k',linewidths=0.5)
        axs[1].set_title("Median")
        axs[2].pcolormesh(lon,lat,mask*grid[2,:,:],shading='auto',vmin=-gridlims,vmax=gridlims,cmap="RdYlBu_r")
        axs[2].contour(lon,lat,mask,[0.5],colors='k',linewidths=0.5)
        axs[2].set_title("Upper bound")
        fig.suptitle('Process '+ process+" Scenario "+scenario + " Year "+str(year))
        fig.colorbar(tst, ax=axs[2], orientation='horizontal', fraction=.075,label="Sea level (m)")
        fig.tight_layout()
        return

    def write_scenario_map(confidence,year,scenario,process):

        # lat,lon coords and land-sea mask
        fn = file
        fh = Dataset(fn, 'r')
        fh.set_auto_mask(False)
        lat = fh['lat'][1030:].reshape(181,360)[:,0]
        lon = fh['lon'][1030:].reshape(181,360)[0,:]
        lon[lon<0]+=360
        fh.close()

        #mask = compute_sea_mask(lat, lon)

        fn = file
        fh = Dataset(fn, 'r')
        fh.set_auto_mask(False)

        # Time steps
        years = fh['years'][:14]
        year_idx = np.where(years == year)[0][0]

        # Find quantiles
        l_quant = np.where(fh['quantiles'][:] == 0.05)[0][0]
        m_quant = np.where(fh['quantiles'][:] == 0.50)[0][0]
        h_quant = np.where(fh['quantiles'][:] == 0.95)[0][0]
        quants = np.array([l_quant,m_quant,h_quant])

        grid = fh['sea_level_change'][quants,year_idx,1030:].reshape(3,181,360)
        fh.close()

        # Write the scenario

        fn = process + "_" + str(year) + '_' + scenario +'_' + confidence +'.nc'
        fh = Dataset(fn, 'w')
        fh.createDimension('lon', len(lon))
        fh.createDimension('lat', len(lat))
        fh.createDimension('CI', 3)
        fh.createVariable('lon', 'f4', ('lon',),zlib=True)[:] = lon
        fh.createVariable('lat', 'f4', ('lat',),zlib=True)[:] = lat
        fh.createVariable('CI', 'f4', ('CI',),zlib=True)[:] = np.array([0.05,0.50,0.95])
        fh.createVariable('sealevel_mm','i4',('CI','lat','lon',),zlib=True)[:] = grid
        #fh.createVariable('mask','i4',('lat','lon',),zlib=True)[:] = mask
        fh.close()
        return

    def compute_sea_mask(lat,lon):
        fn = os.getenv('HOME')+'/Data/GRACE/JPL_mascon/LAND_MASK.CRI.nc'
        fh = Dataset(fn, 'r')
        fh.set_auto_mask(False)
        lon_GRACE = fh["lon"][:]
        lat_GRACE = fh["lat"][:]
        mask_GRACE = 1.0 - fh['land_mask'][:]

        mask = np.flipud((interp2d(lon_GRACE,lat_GRACE,mask_GRACE,kind='linear')(lon,lat)) > 0.5)

        return mask

    # Produce converted SLR files
    #example_run(np.arange(2020,2110,10))

    # Code above are taken from internet

    #Merge downloaded srtm 30m files
    #files_to_mosaic = glob.glob('srtm_30m/*.tif')

    #from osgeo import gdal

    #g = gdal.Warp("ph_srtm30m.tif", files_to_mosaic, format="GTiff",
    #              options=["COMPRESS=LZW", "TILED=YES"]) # if you want
    #g = None #

    ds_srtm = rx.open_rasterio(os.path.join(settings.BASE_DIR, 'sea_level_rise_analysis', 'static', 'input_files', 'ph_srtm30m.tif'))

    # For sample checking of mosaiced/merged SRTM file
    #ax = plt.axes(projection=ccrs.PlateCarree())
    #ds_srtm = ds_srtm.sel(x=slice(120.0,121.0),y=slice(15.0,14.0))
    #ds_srtm.plot(ax=ax)
    #ax.coastlines()
    #plt.savefig(f"srtm30m.png")
    #plt.close('all')
    #sys.exit()

    # For setting buffer of facility area
    #buffer = 0.0009 # degree, ~100 meters radius, ~200 x 200 meters
    #buffer = 0.00225  # degree, ~250 meters radius, ~500 m x 500 m
    #buffer = 0.0045 # degree, ~500 meters radius, ~1km x 1km
    buffer_list = [0.00045, 0.0009, 0.00225, 0.0045, 0.009, 0.0225]
    # buffer_list = [0.00045]

    for buffer in buffer_list:
        buffer_in_km = round(buffer*2*111,1) # convert to km

        # Open facility location and georeference, and instantiate buffer data
        df_fac = pd.read_csv(facility_csv_path)

        print(df_fac.columns)
        # sys.exit()

        crs = {'init':f'EPSG:{4326}'}
        geometry = [Point(xy).buffer(buffer, cap_style = 3) for xy in zip(df_fac['Long'], df_fac['Lat'])]
        geometry_point = [Point(xy) for xy in zip(df_fac['Long'], df_fac['Lat'])]
        geo_df = gpd.GeoDataFrame(df_fac, crs = crs, geometry = geometry)
        geo_df_point = gpd.GeoDataFrame(df_fac, crs = crs, geometry = geometry_point)

        # Open raster for LECZ data
        lecz_path = os.path.join(settings.BASE_DIR, 'sea_level_rise_analysis', 'static', 'input_files', 'merit_leczs.tif')
        ds_lecz = rx.open_rasterio(lecz_path)
        ds_lecz = ds_lecz.sel(x=slice(115.0, 128.0), y=slice(22.0, 3.0))

        # Select lecz 10m and below
        lecz_thresh = 10
        ds_lecz = ds_lecz.where(ds_lecz <= lecz_thresh )

        # save ph filtered merit file and plot for checking
        #ds_lecz.rio.to_raster("merit_lecz_ph.tif")

        #ds_lecz_jg_petro = ds_lecz.sel(x=slice(120.5,121.5),y=slice(14.0,13.0))
        #ds_lecz_mm = ds_lecz.sel(x=slice(120.5,121.3),y=slice(15.0,14.0))

        #ax = plt.axes(projection=ccrs.PlateCarree())
        #ds_lecz_jg_petro.plot(ax=ax)
        #ax.coastlines()
        #ax.scatter(df_fac.long[0], df_fac.lat[0])
        #plt.savefig(f"merit_lecz_jg_plant.png")
        #plt.close('all')

        #ax = plt.axes(projection=ccrs.PlateCarree())
        #ds_lecz_mm.plot(ax=ax)
        #ax.coastlines()
        #plt.savefig(f"merit_lecz_mm.png")
        #plt.close('all')

        # Open merit LECZ ph file
        import rasterio
        from rasterio.features import shapes
        mask = None
        with rasterio.Env():
            file_path = os.path.join(settings.BASE_DIR, 'sea_level_rise_analysis', 'static', 'input_files', 'merit_lecz_ph.tif')
            with rasterio.open(file_path) as src:
                image = src.read(1)  # first band
                image = np.float32(image)
                results = (
                    {'properties': {'raster_val': v}, 'geometry': s}
                    for i, (s, v) in enumerate(shapes(image, mask=mask, transform=src.transform))
                )

        geoms = list(results)
        # check first feature
        print (geoms[0])

        #convert lecz raster to shapefile
        from shapely.geometry import shape
        print (shape(geoms[0]['geometry']))
        import geopandas as gp
        lecz_shp  = gp.GeoDataFrame.from_features(geoms).set_crs(epsg=4326)
        lecz_shp = lecz_shp[lecz_shp.raster_val.notnull()]
        print (lecz_shp)

        # plot shapefile for checking
        #ax = plt.axes(projection=ccrs.PlateCarree())
        #lecz_shp.plot(ax=ax, color='lightgrey')
        #plt.savefig(f"merit_lecz_shapefile.png")
        #plt.close('all')

        # Spatial join to check overlap between buffer/point facility locations and lecz shapefile
        df_sjoin = sjoin(geo_df, lecz_shp, how='inner', predicate='intersects')
        df_sjoin = df_sjoin[["Site", "Lat", "Long", "geometry"]]
        df_sjoin_point = geo_df_point[geo_df_point.index.isin(df_sjoin.index)]
        df_sjoin_point = df_sjoin_point.drop_duplicates(
        subset = ['Lat', 'Long'],
        keep = 'first').reset_index(drop = True)
        df_sjoin_point_latlon= df_sjoin_point[["Site","Long","Lat"]]
        #print(df_sjoin_point)
        #print(df_sjoin)

        # Plot lecz map and facilities
        fig, ax = plt.subplots(figsize = (10,10))
        lecz_shp.plot(ax=ax, color='lightgrey')
        df_sjoin_point.plot(ax=ax)
        ax.set_title(f'Low elevation coastal zone ({lecz_thresh}m) map,\nbuffer zone = {buffer_in_km} x {buffer_in_km} km,\n{len(df_sjoin_point)} facilities')
        ax.figure.savefig(f"lecz_{lecz_thresh}m_buffer{buffer}deg_merit.png")
        plt.close('all')

        # get SLR projetions for facility locations, different future decades, and projection quantiles
        quant_list = [0.05, 0.5 ,0.95]
        year_list = np.arange(2030,2061,10)

        for i, year in enumerate(year_list):
            for j, quant in enumerate(quant_list):
                # Sea level rise projections and SRTM DEM
                ds_slr = xr.open_dataset(os.path.join(settings.BASE_DIR, 'sea_level_rise_analysis', 'static', 'input_files', f"total_{year}_ssp585_Medium.nc"))
                ds_slr = ds_slr.sel(lon=slice(115.0,128.0),lat=np.arange(3,22))
                #print(np.max(ds_slr.sealevel_mm.values))

                # select quantile and remove missing values
                ds_slr = ds_slr.sel(CI=quant)
                ds_slr = ds_slr.where(ds_slr['sealevel_mm'] > -32768)  

                ## plot sea level rise
                #ax = plt.axes(projection=ccrs.PlateCarree())
                #p = ds_slr.sealevel_mm.plot(ax=ax)
        
                #ax.scatter(df_sjoin_point_latlon.long,df_sjoin_point_latlon.lat, s=1)
                #ax.axes.coastlines()
                #plt.savefig(f"total_{year}_ssp585_Medium_quant{quant}_PH_CI{quant}.png")
                #plt.close('all')
                
                # create dataframe for saving CSV output file
                quant_col = f"{year} Sea Level Rise CI {quant}"
                cols = ["Facility", "Lat", "Lon", "SRTM elevation", quant_col]
                df_quant = pd.DataFrame(columns=cols)

                #print(df_sjoin_point_latlon)

                # Loop through facility locations
                for row in df_sjoin_point_latlon.itertuples():
                    print(row)

                    # get minimum value in srtm search area based on facility buffer size
                    long_min = row.Long - buffer
                    long_max = row.Long + buffer
                    lat_min = row.Lat - buffer
                    lat_max = row.Lat + buffer

                    # Different quantiles for selecting elevation value within faciliy buffer region
                    #srtm_elev = ds_srtm.sel(x=slice(long_min, long_max),  y=slice(lat_max, lat_min)).values.mean()
                    #srtm_elev = ds_srtm.sel(x=slice(long_min, long_max),  y=slice(lat_max, lat_min)).values.min()
                    #srtm_elev = ds_srtm.sel(x=slice(long_min, long_max),  y=slice(lat_max, lat_min)).quantile(0.1).values
                    srtm_elev = ds_srtm.sel(x=slice(long_min, long_max),  y=slice(lat_max, lat_min)).quantile(0.05).round().values
                    #srtm_elev = ds_srtm.sel(x=slice(long_min, long_max),  y=slice(lat_max, lat_min)).quantile(0.5).values
                    #srtm_elev = ds_srtm.sel(x=row.long, y=row.lat, method="nearest").values[0] 
                    slr_proj = ds_slr.sel(lon=row.Long, lat=row.Lat, method="nearest").sealevel_mm.values / 1000 # convert to meters

                    # filter out > 10m elevation, in case there is discrepancy between merit lecz and srtm
                    if srtm_elev <= 10:
                        df_temp = pd.DataFrame([[row.Site, row.Lat, row.Long, srtm_elev, slr_proj]], columns=cols)
                        df_quant = pd.concat([df_quant, df_temp])

                # for very first entry, set dataframe
                if i==0 and j==0:
                    df_master = df_quant
                # else concat column-wise
                else:
                    df_master = pd.concat([df_master, df_quant[quant_col]], axis=1)
            
        # Save to csv    
        print(df_master)
        df_master.to_csv(f"JG_slr_allYears_lecz{lecz_thresh}m_buffer{buffer_in_km}km_allCI_meritLECZ_srtm30_srtmQuant0.05Search.csv", index=False)