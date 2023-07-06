# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Forcing data

# %% [markdown]
# In this notebook we will download ERA5 forcing data for the catchment that was determined in the previous notebook *Catchment delineation*. This includes
#
# 1. ... the geopotential height 
# 2. ... the temperature and precipitation Time Series
#
# We will use Google Earth Engine (GEE) to retrieve the ERA5-Land data and to perform spatial calculations. 
#
# > ERA5-Land is a reanalysis dataset providing a consistent view of the evolution of land variables over several decades at an enhanced resolution compared to ERA5. ERA5-Land has been produced by replaying the land component of the ECMWF ERA5 climate reanalysis. Reanalysis combines model data with observations from across the world into a globally complete and consistent dataset using the laws of physics. Reanalysis produces data that goes several decades back in time, providing an accurate description of the climate of the past. 
# >
# > Source: https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_RAW#description

# %% [markdown]
# Let's start by importing required packages and defining functions.

# %%
# Google Earth Engine packages
import ee
import geemap
import geemap.colormaps as cm

import xarray as xr
import numpy as np


# function to load nc file into GEE
def netcdf_to_ee(ds):
    data = ds['z']

    lon_data = np.round(data['lon'], 3)
    lat_data = np.round(data['lat'], 3)

    dim_lon = np.unique(np.ediff1d(lon_data).round(3))
    dim_lat = np.unique(np.ediff1d(lat_data).round(3))

    if (len(dim_lon) != 1) or (len(dim_lat) != 1):
        print("The netCDF file is not a regular longitude/latitude grid")

    data_np = np.array(data)
    data_np = np.transpose(data_np)

    # Figure out if we need to roll the data or not
    # (see https://github.com/giswqs/geemap/issues/285#issuecomment-791385176)
    if np.max(lon_data) > 180:
        data_np = np.roll(data_np, 180, axis=0)
        west_lon = lon_data[0] - 180
    else:
        west_lon = lon_data[0]

    transform = [dim_lon[0], 0, float(west_lon) - dim_lon[0]/2, 0, dim_lat[0], float(lat_data[0]) - dim_lat[0]/2]

    image = geemap.numpy_to_ee(
        data_np, "EPSG:4326", transform=transform, band_names='z'
    )
    return image, data_np, transform


# %% [markdown]
# First of all, the Google Earth Engine (GEE) access must be initialized. When using it for the first time on this machine, you need to authenticate first. A more detailled explanation on how to authenticate is provided in the first notebook.

# %%
# initialize GEE at the beginning of session
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()         # authenticate when using GEE for the first time
    ee.Initialize()

# %% [markdown]
# New read from the `config.ini` file which is used throughout the different notebooks:
#
# - input/output folders for data imports and downloads
# - filenames (DEM, GeoPackage)
# - past modeling vs. future projections
# - show/hide GEE map in notebooks

# %%
import pandas as pd
import configparser
import ast

# read local config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

# get file config from config.ini
dir_input = config['FILE_SETTINGS']['DIR_INPUT']
dir_output = config['FILE_SETTINGS']['DIR_OUTPUT']
output_gpkg = dir_output + config['FILE_SETTINGS']['GPKG_NAME']
scenarios = config.getboolean('CONFIG', 'PROJECTIONS')
show_map = config.getboolean('CONFIG','SHOW_MAP')

# %% [markdown]
# Load the catchment outline which was stored as a result of the previous notebook and convert it to a GEE feature collection so it can be added to the map later.

# %%
import geopandas as gpd

catchment_new = gpd.read_file(output_gpkg, layer='catchment_new')
catchment = geemap.geopandas_to_ee(catchment_new)

# %% [markdown]
# ## Configuration of date range for downloading

# %% [markdown]
# If you are only interested in modeling the past, set `PROJECTIONS=False` in the `config.ini` to download reanalysis data for your defined modeling period only. Otherwise, all available historic data (since 1979) is downloaded to provide the best possible basis for bias adjustment of the climate scenario data.

# %%
if scenarios == True:
    date_range = ['1979-01-01', '2023-01-01']
else:
    date_range = ast.literal_eval(config['CONFIG']['DATE_RANGE'])

print(f'The selected date range is from {date_range[0]} to {date_range[1]}')

# %% [markdown]
# ***

# %% [markdown]
# ## ERA5L Geopotential height

# %% [markdown]
# > This parameter is the gravitational potential energy of a unit mass, at a particular location, relative to mean sea level. It is also the amount of work that would have to be done, against the force of gravity, to lift a unit mass to that location from mean sea level.
# >
# > The geopotential height can be calculated by dividing the geopotential by the Earth's gravitational acceleration, g (=9.80665 m s-2). The geopotential height plays an important role in synoptic meteorology (analysis of weather patterns). Charts of geopotential height plotted at constant pressure levels (e.g., 300, 500 or 850 hPa) can be used to identify weather systems such as cyclones, anticyclones, troughs and ridges.
# >
# > At the surface of the Earth, this parameter shows the variations in geopotential (height) of the surface, and is often referred to as the orography.
# >
# > Source: https://codes.ecmwf.int/grib/param-db/?id=129

# %% [markdown]
# Since the ERA5 geopotential height is not yet available at the Google Earth Engine Data Catalog, the file must be downloaded from a media server with specific references. The login data and API key must be defined in the `config.ini` file.

# %%
from resourcespace import ResourceSpace

# use guest credentials to access media server 
api_base_url = config['MEDIA_SERVER']['api_base_url']
private_key = config['MEDIA_SERVER']['private_key']
user = config['MEDIA_SERVER']['user']

myrepository = ResourceSpace(api_base_url, user, private_key)

# get resource IDs for each .zip file
refs_era5l = pd.DataFrame(myrepository.get_collection_resources(128))[['ref', 'file_size', 'file_extension', 'field8']]
ref_geopot = refs_era5l.loc[refs_era5l['field8'] == 'ERA5_land_Z_geopotential']
print("Dataset file and reference on media server:")
display(ref_geopot)

# %% [markdown]
# Now extract the `.zip` file and load the `NetCDF` file as `xarray` dataset for further processing.

# %%
from zipfile import ZipFile
import io

content = myrepository.get_resource_file(ref_geopot.at[0,'ref'])    
with ZipFile(io.BytesIO(content), 'r') as zipObj:
    # Get a list of all archived file names from the zip
    filename = zipObj.namelist()[0]
    print(f'Reading file "{filename}"...')
    file_bytes = zipObj.read(filename)

# Open the file-like object as an xarray dataset
ds = xr.open_dataset(io.BytesIO(file_bytes))
print(f'Dataset contains {ds.z.attrs["long_name"]} in {ds.z.attrs["units"]} as variable \'{ds.z.attrs["GRIB_cfVarName"]}\'')

# %% [markdown]
# The dataset convers the entire globe. In order to increase runtime, the dataset will be cropped to the catchment area with a 1° buffer zone.

# %%
# get catchment bounding box and add 1° buffer before cropping geopotential dataset
bounds = catchment_new.total_bounds
min_lon = bounds[0] - 1
min_lat = bounds[1] - 1
max_lon = bounds[2] + 1
max_lat = bounds[3] + 1

cropped_ds = ds.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon))

# %% [markdown]
# Transform the `NetCDF` data so it can be added and processed within GEE.

# %%
image, data_np, transform = netcdf_to_ee(cropped_ds)

# %% [markdown]
# Now we are ready to go. Let's start with the base map if enabled in `config.ini`. In the first step, the geopotential height and the catchment outline will be added. <a id="map"></a>

# %%
if show_map:
    Map = geemap.Map()
    # add geopotential as layer
    vis_params =  {'min': int(data_np.min()), 'max': int(data_np.max()), 'palette': cm.palettes.terrain, 'opacity': 0.8}
    Map.addLayer(image, vis_params, "ERA5L geopotential")
    
    # add catchment
    Map.addLayer(catchment, {'color': 'darkgrey'}, "Catchment")
    Map.centerObject(catchment, zoom=9)
    display(Map)
else:
    print("Map view disabled in config.ini")        

# %% [markdown]
# As a last step, we need to calculate the weighted average geopotential for the catchment area and convert it to elevation in meters above sea level. 

# %%
# execute reducer
dict = image.reduceRegion(ee.Reducer.mean(),
                          geometry=catchment,
                          crs='EPSG:4326',
                          crsTransform=transform)

# get mean value and print
mean_val = dict.getInfo()['z']
ele_dat = mean_val / 9.80665
print(f'Geopotential mean:\t{mean_val:.2f} m2 s-2\nElevation:\t\t {ele_dat:.2f} m a.s.l.')

# %% [markdown]
# ***

# %% [markdown]
# ## ERA5L Temperature and Precipitation Time Series

# %% [markdown]
# For the temperature and precipitation time series data, we will use the **ERA5-Land Daily Aggregated - ECMWF Climate Reanalysis** `ECMWF/ERA5_LAND/DAILY_RAW` dataset from the <a href="https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_RAW#bands">Google Earth Engine Data Catalog</a>
#
# > The asset is a daily aggregate of ECMWF ERA5 Land hourly assets. [...] Daily aggregates have been pre-calculated to facilitate many applications requiring easy and fast access to the data.
# >
# > Source: https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_RAW#description

# %% [markdown]
# Load the time series dataset from GEE and
#
# - ... apply the defined date range
# - ... only select the relevant bands for temperature and precipitation
# - ... get `mean` values for the catchment area

# %%
import pandas as pd
import datetime

def setProperty(image):
    dict = image.reduceRegion(ee.Reducer.mean(), catchment)
    return image.set(dict)


collection = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_RAW')\
    .select('temperature_2m','total_precipitation_sum')\
    .filterDate(date_range[0], date_range[1])

withMean = collection.map(setProperty)

# %% [markdown]
# Now convert the results into a dataframe for easy downloading. This might take some time depending on the selected date range.

# %%
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

# %% [markdown]
# Show content of dataframe:

# %%
display(df)

# %% [markdown]
# Create a plot that shows the determined ERA5-Land data for defined time series for temperature and precipitation.

# %%
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

axes = df.drop(['ts','temp'],axis=1).plot.line(x='dt', subplots=True, legend=False, figsize=(10,5),
                                               title='ERA5-Land Data for Catchment',
                                               color={"temp_c": "red", "prec": "blue"})
axes[0].set_ylabel("Temperature [°C]")
axes[1].set_ylabel("Precipitation [mm]")
axes[1].set_xlabel("Date")
axes[1].xaxis.set_minor_locator(mdates.YearLocator())
plt.xlim(date_range)
plt.tight_layout()
plt.show()

# %% [markdown]
# # Store values for other notebooks

# %% [markdown]
# Export ERA5 data to `.csv` file for later processing. 

# %%
df.to_csv(dir_output + 'ERA5L.csv',header=True,index=False)

# %% [markdown]
# Update `settings.yml` file and store the relevant catchment information. Those information will be used in later notebooks:
#
# - **ele_dat**: ERA5 reference elevation for catchment

# %%
import yaml

def read_yaml(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
        return data
    
def write_yaml(data, file_path):
    with open(file_path, 'w') as f:
        yaml.safe_dump(data, f)

def update_yaml(file_path, new_items):
    data = read_yaml(file_path)
    data.update(new_items)
    write_yaml(data, file_path)

        
update_yaml(dir_output + 'settings.yml', {'ele_dat': float(ele_dat)})

# TEST Change for pre-commit hook

# %%
# %reset -f

