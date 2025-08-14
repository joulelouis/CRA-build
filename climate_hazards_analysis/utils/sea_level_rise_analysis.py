


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
    try:
        # Lists to track generated files
        output_csv_files = []
        output_png_files = []

        file = "total_ssp585_medium_confidence_values.nc"

        # (Definitions for example_run, plot_scenario_map, write_scenario_map, compute_sea_mask remain unchanged)
        def example_run(years):
            confidence = 'Medium'
            scenario = 'ssp585'
            process = 'total'
            for yr in years:
                write_scenario_map(confidence, yr, scenario, process)

        def plot_scenario_map(confidence, year, scenario, process):
            fn = file
            fh = Dataset(fn, 'r')
            fh.set_auto_mask(False)
            lat = fh['lat'][1030:].reshape(181, 360)[:, 0]
            lon = fh['lon'][1030:].reshape(181, 360)[0, :]
            lon[lon < 0] += 360
            fh.close()

            mask = compute_sea_mask(lat, lon)

            fn = file
            fh = Dataset(fn, 'r')
            fh.set_auto_mask(False)
            years = fh['years'][:14]
            year_idx = np.where(years == year)[0][0]
            l_quant = np.where(fh['quantiles'][:] == 0.05)[0][0]
            m_quant = np.where(fh['quantiles'][:] == 0.50)[0][0]
            h_quant = np.where(fh['quantiles'][:] == 0.95)[0][0]
            quants = np.array([l_quant, m_quant, h_quant])
            fh[quants, year_idx, 1030:].reshape(3, 181, 360) / 1000
            fh.close()

            gridlims = 0.9 * np.max(grid)
            fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
            tst = axs[0].pcolormesh(lon, lat, mask * grid[0, :, :], shading='auto',
                                    vmin=-gridlims, vmax=gridlims, cmap="RdYlBu_r")
            axs[0].set_title("Lower bound")
            axs[0].contour(lon, lat, mask, [0.5], colors='k', linewidths=0.5)
            axs[1].pcolormesh(lon, lat, mask * grid[1, :, :], shading='auto',
                                    vmin=-gridlims, vmax=gridlims, cmap="RdYlBu_r")
            axs[1].contour(lon, lat, mask, [0.5], colors='k', linewidths=0.5)
            axs[1].set_title("Median")
            axs[2].pcolormesh(lon, lat, mask * grid[2, :, :], shading='auto',
                                    vmin=-gridlims, vmax=gridlims, cmap="RdYlBu_r")
            axs[2].contour(lon, lat, mask, [0.5], colors='k', linewidths=0.5)
            axs[2].set_title("Upper bound")
            fig.suptitle('Process ' + process + " Scenario " + scenario + " Year " + str(year))
            fig.colorbar(tst, ax=axs[2], orientation='horizontal', fraction=.075, label="Sea level (m)")
            fig.tight_layout()
            return

        def write_scenario_map(confidence, year, scenario, process):
            fn = file
            fh = Dataset(fn, 'r')
            fh.set_auto_mask(False)
            lat = fh['lat'][1030:].reshape(181, 360)[:, 0]
            lon = fh['lon'][1030:].reshape(181, 360)[0, :]
            lon[lon < 0] += 360
            fh.close()

            fn = file
            fh = Dataset(fn, 'r')
            fh.set_auto_mask(False)
            years = fh['years'][:14]
            year_idx = np.where(years == year)[0][0]
            l_quant = np.where(fh['quantiles'][:] == 0.05)[0][0]
            m_quant = np.where(fh['quantiles'][:] == 0.50)[0][0]
            h_quant = np.where(fh['quantiles'][:] == 0.95)[0][0]
            quants = np.array([l_quant, m_quant, h_quant])
            grid = fh['sea_level_change'][quants, year_idx, 1030:].reshape(3, 181, 360)
            fh.close()

            # Write out the scenario as a new netCDF file.
            fn_out = process + "_" + str(year) + '_' + scenario + '_' + confidence + '.nc'
            fh = Dataset(fn_out, 'w')
            fh.createDimension('lon', len(lon))
            fh.createDimension('lat', len(lat))
            fh.createDimension('CI', 3)
            fh.createVariable('lon', 'f4', ('lon',), zlib=True)[:] = lon
            fh.createVariable('lat', 'f4', ('lat',), zlib=True)[:] = lat
            fh.createVariable('CI', 'f4', ('CI',), zlib=True)[:] = np.array([0.05, 0.50, 0.95])
            fh.createVariable('sealevel_mm', 'i4', ('CI', 'lat', 'lon',), zlib=True)[:] = grid
            fh.close()
            return

        def compute_sea_mask(lat, lon):
            fn = os.getenv('HOME') + '/Data/GRACE/JPL_mascon/LAND_MASK.CRI.nc'
            fh = Dataset(fn, 'r')
            fh.set_auto_mask(False)
            lon_GRACE = fh["lon"][:]
            lat_GRACE = fh["lat"][:]
            mask_GRACE = 1.0 - fh['land_mask'][:]
            mask = np.flipud((interp2d(lon_GRACE, lat_GRACE, mask_GRACE, kind='linear')(lon, lat)) > 0.5)
            return mask

        # Load the SRTM raster using an absolute path.
        ds_srtm = rx.open_rasterio(os.path.join(
            settings.BASE_DIR,
            'sea_level_rise_analysis',
            'static',
            'input_files',
            'ph_srtm30m.tif'
        ))

        # Define buffer sizes to iterate through.
        # buffer_list = [0.00045, 0.0009, 0.00225, 0.0045, 0.009, 0.0225]
        buffer_list = [0.00045]

        for buffer in buffer_list:
            buffer_in_km = round(buffer * 2 * 111, 1)  # convert to km

            # Open facility CSV and create geodataframes.
            df_fac = pd.read_csv(facility_csv_path)

            # Standardize expected column names
            rename_map = {}
            for col in df_fac.columns:
                low = col.strip().lower()
                if low in ['facility', 'site', 'site name', 'facility name', 'facilty name', 'name', 'asset name']:
                    rename_map[col] = 'Facility'
                elif low == 'latitude' and 'Lat' not in df_fac.columns:
                    rename_map[col] = 'Lat'
                elif low == 'longitude' and 'Long' not in df_fac.columns:
                    rename_map[col] = 'Long'
            if rename_map:
                df_fac.rename(columns=rename_map, inplace=True)

            required_cols = ['Facility', 'Lat', 'Long']
            missing = [c for c in required_cols if c not in df_fac.columns]
            if missing:
                raise ValueError(f"Missing required columns in facility CSV: {', '.join(missing)}")

            crs = {'init': f'EPSG:{4326}'}
            geometry = [Point(xy).buffer(buffer, cap_style=3) for xy in zip(df_fac['Long'], df_fac['Lat'])]
            geometry_point = [Point(xy) for xy in zip(df_fac['Long'], df_fac['Lat'])]
            geo_df = gpd.GeoDataFrame(df_fac, crs=crs, geometry=geometry)
            geo_df_point = gpd.GeoDataFrame(df_fac, crs=crs, geometry=geometry_point)

            # Load LECZ data using an absolute path.
            lecz_path = os.path.join(
                settings.BASE_DIR,
                'sea_level_rise_analysis',
                'static',
                'input_files',
                'merit_leczs.tif'
            )
            ds_lecz = rx.open_rasterio(lecz_path)
            ds_lecz = ds_lecz.sel(x=slice(115.0, 128.0), y=slice(22.0, 3.0))
            lecz_thresh = 10
            ds_lecz = ds_lecz.where(ds_lecz <= lecz_thresh)

            # Open the processed LECZ raster for vectorization.
            import rasterio
            from rasterio.features import shapes
            mask = None
            with rasterio.Env():
                file_path = os.path.join(
                    settings.BASE_DIR,
                    'sea_level_rise_analysis',
                    'static',
                    'input_files',
                    'merit_lecz_ph.tif'
                )
                with rasterio.open(file_path) as src:
                    image = src.read(1)  # first band
                    image = np.float32(image)
                    results = (
                        {'properties': {'raster_val': v}, 'geometry': s}
                        for i, (s, v) in enumerate(shapes(image, mask=mask, transform=src.transform))
                    )
            geoms = list(results)
            print(geoms[0])
            from shapely.geometry import shape
            print(shape(geoms[0]['geometry']))
            import geopandas as gp
            lecz_shp = gp.GeoDataFrame.from_features(geoms).set_crs(epsg=4326)
            lecz_shp = lecz_shp[lecz_shp.raster_val.notnull()]
            print(lecz_shp)

            # Spatial join between facility buffers and the LECZ shapefile.
            df_sjoin = sjoin(geo_df, lecz_shp, how='inner', predicate='intersects')
            df_sjoin = df_sjoin[["Facility", "Lat", "Long", "geometry"]]
            df_sjoin_point = geo_df_point[geo_df_point.index.isin(df_sjoin.index)]
            df_sjoin_point = df_sjoin_point.drop_duplicates(
                subset=['Lat', 'Long'],
                keep='first').reset_index(drop=True)
            df_sjoin_point_latlon = df_sjoin_point[["Facility", "Long", "Lat"]]

            # Plot the LECZ map with facilities and save the PNG.
            fig, ax = plt.subplots(figsize=(10, 10))
            lecz_shp.plot(ax=ax, color='lightgrey')
            df_sjoin_point.plot(ax=ax)
            ax.set_title(f'Low elevation coastal zone ({lecz_thresh}m) map,\n'
                         f'buffer zone = {buffer_in_km} x {buffer_in_km} km,\n'
                         f'{len(df_sjoin_point)} facilities')
            output_png = os.path.join(
                settings.BASE_DIR,
                'sea_level_rise_analysis',
                'static',
                'input_files',
                f"lecz_{lecz_thresh}m_buffer{buffer}deg_merit.png"
            )
            ax.figure.savefig(output_png)
            output_png_files.append(output_png)
            plt.close('all')

            # Get SLR projections for facility locations.
            quant_list = [0.05, 0.5, 0.95]
            year_list = np.arange(2030, 2061, 10)

            for i, year in enumerate(year_list):
                for j, quant in enumerate(quant_list):
                    ds_slr = xr.open_dataset(os.path.join(
                        settings.BASE_DIR,
                        'sea_level_rise_analysis',
                        'static',
                        'input_files',
                        f"total_{year}_ssp585_Medium.nc"
                    ))
                    ds_slr = ds_slr.sel(lon=slice(115.0, 128.0), lat=np.arange(3, 22))
                    ds_slr = ds_slr.sel(CI=quant)
                    ds_slr = ds_slr.where(ds_slr['sealevel_mm'] > -32768)

                    quant_col = f"{year} Sea Level Rise CI {quant}"
                    cols = ["Facility", "Lat", "Long", "SRTM elevation", quant_col]
                    df_quant = pd.DataFrame(columns=cols)

                    for row in df_sjoin_point_latlon.itertuples():
                        print(row)
                        long_min = row.Long - buffer
                        long_max = row.Long + buffer
                        lat_min = row.Lat - buffer
                        lat_max = row.Lat + buffer

                        srtm_elev = ds_srtm.sel(x=slice(long_min, long_max),
                                                y=slice(lat_max, lat_min)).quantile(0.05).round().values
                        slr_proj = ds_slr.sel(lon=row.Long, lat=row.Lat, method="nearest").sealevel_mm.values / 1000
                        if srtm_elev <= 10:
                            df_temp = pd.DataFrame([[row.Facility, row.Lat, row.Long, srtm_elev, slr_proj]], columns=cols)
                            df_quant = pd.concat([df_quant, df_temp])
                    if i == 0 and j == 0:
                        df_master = df_quant
                    else:
                        df_master = pd.concat([df_master, df_quant[quant_col]], axis=1)
            
            # Save the master CSV file and track its path.
            output_csv = os.path.join(
                settings.BASE_DIR,
                'sea_level_rise_analysis',
                'static',
                'input_files',
                f"JG_slr_allYears_lecz{lecz_thresh}m_buffer{buffer_in_km}km_allCI_meritLECZ_srtm30_srtmQuant0.05Search.csv"
            )
            print(df_master)
            df_master.to_csv(output_csv, index=False)
            output_csv_files.append(output_csv)

        # Return a dictionary with all output file paths.
        return {
            "combined_csv_paths": output_csv_files,
            "png_paths": output_png_files
        }

    except Exception as e:
        return {"error": str(e)}
