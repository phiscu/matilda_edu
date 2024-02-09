import ee
import pandas as pd
import configparser
import ast
import geopandas as gpd
import geemap
from IPython.core.display_functions import display
from zipfile import ZipFile
import io
import xarray as xr
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os
# Add change cwd to matilda_edu home dir and add it to PATH
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.append(cwd)            # Hotfix. Master script runs subprocess that changes the CWD.
sys.path.append(parent_dir)
from resourcespace import ResourceSpace
from tools.helpers import update_yaml

## Initialize GEE
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()  # authenticate when using GEE for the first time
    ee.Initialize()

# read local config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

# get file config from config.ini
dir_input = config['FILE_SETTINGS']['DIR_INPUT']
dir_output = config['FILE_SETTINGS']['DIR_OUTPUT']
output_gpkg = dir_output + config['FILE_SETTINGS']['GPKG_NAME']
scenarios = config.getboolean('CONFIG', 'PROJECTIONS')

#  load the catchment outline and convert it to a `ee.FeatureCollection`
catchment_new = gpd.read_file(output_gpkg, layer='catchment_new')
catchment = geemap.geopandas_to_ee(catchment_new)

# Set the date range
if scenarios == True:
    date_range = ['1979-01-01', '2023-01-01']
else:
    date_range = ast.literal_eval(config['CONFIG']['DATE_RANGE'])

print(f'The selected date range is {date_range[0]} to {date_range[1]}')

## Derive ERA5L Geopotential height

# use guest credentials to access media server
api_base_url = config['MEDIA_SERVER']['api_base_url']
private_key = config['MEDIA_SERVER']['private_key']
user = config['MEDIA_SERVER']['user']

myrepository = ResourceSpace(api_base_url, user, private_key)

# get resource IDs for each .zip file
refs_era5l = pd.DataFrame(myrepository.get_collection_resources(128))[['ref', 'file_size', 'file_extension', 'field8']]
ref_geopot = refs_era5l.loc[refs_era5l['field8'] == 'ERA5_land_Z_geopotential']
print("Dataset file and reference on media server:\n")
display(ref_geopot)

# Unzip geopotential `.ncdf` and load it as `xarray` dataset for further processing.
content = myrepository.get_resource_file(ref_geopot.at[0, 'ref'])
with ZipFile(io.BytesIO(content), 'r') as zipObj:
    # Get a list of all archived file names from the zip
    filename = zipObj.namelist()[0]
    print(f'Reading file "{filename}"...')
    file_bytes = zipObj.read(filename)

# Open the file-like object as an xarray dataset
ds = xr.open_dataset(io.BytesIO(file_bytes))
print(
    f'Dataset contains {ds.z.attrs["long_name"]} in {ds.z.attrs["units"]} as variable \'{ds.z.attrs["GRIB_cfVarName"]}\'')

# crop original file to catchment area plus a 1° buffer zone.
bounds = catchment_new.total_bounds
min_lon = bounds[0] - 1
min_lat = bounds[1] - 1
max_lon = bounds[2] + 1
max_lat = bounds[3] + 1

cropped_ds = ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
print(f"xr.Dataset cropped to bbox[{round(min_lon, 2)}, {round(min_lat, 2)}, {round(max_lon, 2)}, {round(max_lat)}]")

# load `xarray` data into GEE (workaround) --> by Oliver Lopez (https://github.com/lopezvoliver/geemap/blob/netcdf_to_ee/geemap/common.py#L1776).

# function to load nc file into GEE
def netcdf_to_ee(ds):
    data = ds['z']

    lon_data = np.round(data['lon'], 3)
    lat_data = np.round(data['lat'], 3)

    dim_lon = np.unique(np.ediff1d(lon_data).round(3))
    dim_lat = np.unique(np.ediff1d(lat_data).round(3))

    if (len(dim_lon) != 1) or (len(dim_lat) != 1):
        print("The netCDF file is not a regular longitude/latitude grid")

    print("Converting xarray to numpy array...")
    data_np = np.array(data)
    data_np = np.transpose(data_np)

    # Figure out if we need to roll the data or not
    # (see https://github.com/giswqs/geemap/issues/285#issuecomment-791385176)
    if np.max(lon_data) > 180:
        data_np = np.roll(data_np, 180, axis=0)
        west_lon = lon_data[0] - 180
    else:
        west_lon = lon_data[0]

    print("Saving data extent and origin...")
    transform = [dim_lon[0], 0, float(west_lon) - dim_lon[0] / 2, 0, dim_lat[0], float(lat_data[0]) - dim_lat[0] / 2]

    print("Converting numpy array to ee.Array...")
    image = geemap.numpy_to_ee(
        data_np, "EPSG:4326", transform=transform, band_names='z'
    )
    print("Done!")
    return image, data_np, transform

image, data_np, transform = netcdf_to_ee(cropped_ds)

cropped_ds.close()

# Calculate area-weighted average geopotential and convert it to geopotential height in m.a.s.l.
# execute reducer
dict = image.reduceRegion(ee.Reducer.mean(),
                          geometry=catchment,
                          crs='EPSG:4326',
                          crsTransform=transform)

# get mean value and print
mean_val = dict.getInfo()['z']
ele_dat = mean_val / 9.80665
print(f'Geopotential mean:\t{mean_val:.2f} m2 s-2\nElevation:\t\t {ele_dat:.2f} m a.s.l.')

## ERA5-Land Temperature and Precipitation Data

# create an `ee.ImageCollection` with temperature and precipitation bands and date range.
# apply `ee.Reducer` function for area-weighted aggregates.
def setProperty(image):
    dict = image.reduceRegion(ee.Reducer.mean(), catchment)
    return image.set(dict)

collection = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_RAW') \
    .select('temperature_2m', 'total_precipitation_sum') \
    .filterDate(date_range[0], date_range[1])

withMean = collection.map(setProperty)

# aggregate results into arrays, download  with `.getInfo()` and store as dataframe columns.
# %%time
df = pd.DataFrame()
print("Get timestamps...")
df['ts'] = withMean.aggregate_array('system:time_start').getInfo()
df['dt'] = df['ts'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000))
print("Get temperature values...")
df['temp'] = withMean.aggregate_array('temperature_2m').getInfo()
df['temp_c'] = df['temp'] - 273.15
print("Get precipitation values...")
df['prec'] = withMean.aggregate_array('total_precipitation_sum').getInfo()
df['prec'] = df['prec'] * 1000

display(df)

# plot time series
axes = df.drop(['ts', 'temp'], axis=1).plot.line(x='dt', subplots=True, legend=False, figsize=(10, 5),
                                                 title='ERA5-Land Data for Target Catchment',
                                                 color={"temp_c": "red", "prec": "darkblue"})
axes[0].set_ylabel("Temperature [°C]")
axes[1].set_ylabel("Precipitation [mm]")
axes[1].set_xlabel("Date")
axes[1].xaxis.set_minor_locator(mdates.YearLocator())
plt.xlim(date_range)
plt.tight_layout()
plt.savefig(dir_output + 'era5l_raw.png')

# Store ERA5 data in `.csv` file.
df.to_csv(dir_output + 'ERA5L.csv', header=True, index=False)
print("Stored data in 'ERA5L.csv'")

# update `settings.yml` file with the reference altitude of the ERA5-Land data (`ele_dat`).
update_yaml(dir_output + 'settings.yml', {'ele_dat': float(ele_dat)})
print("Updated 'settings.yml'")

