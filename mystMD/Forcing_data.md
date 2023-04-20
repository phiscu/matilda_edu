---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Initialization

```{code-cell} ipython3
# Google Earth Engine packages
import ee
import geemap
import geemap.colormaps as cm

import xarray as xr
import numpy as np


# function to load nc file into GEE
def netcdf_to_ee(filename):
    ds = xr.open_dataset(filename)
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
```

```{code-cell} ipython3
# initialize GEE at the beginning of session
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()         # authenticate when using GEE for the first time
    ee.Initialize()
```

```{code-cell} ipython3
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
```

```{code-cell} ipython3
if show_map:
    Map = geemap.Map()
```

# Configure the downloaded date range

+++

If you are only interested in modeling the past, set `PROJECTIONS=False` in the `config.ini` to download reanalysis data for your defined modeling period only. Otherwise, all available historic data (since 1979) is downloaded to provide the best possible basis for bias adjustment of the climate scenario data.

```{code-cell} ipython3
if scenarios == True:
    date_range = ['1979-01-01', '2022-01-01']
else:
    date_range = ast.literal_eval(config['CONFIG']['DATE_RANGE'])
```

***

+++

# ERA5L Geopotential

+++

### Load input data

+++

Load ERA5L file for geopotential and add to Map

```{code-cell} ipython3
file = 'ERA5_land_Z_geopotential_HMA.nc'
filename = dir_input + file
```

```{code-cell} ipython3
image, data_np, transform = netcdf_to_ee(filename)
```

```{code-cell} ipython3
if show_map:
    # add image as layer
    vis_params =  {'min': int(data_np.min()), 'max': int(data_np.max()), 'palette': cm.palettes.terrain, 'opacity': 0.8}
    Map.addLayer(image, vis_params, "ERA5L geopotential")
```

Load catchment and add to map

```{code-cell} ipython3
import geopandas as gpd

catchment_new = gpd.read_file(output_gpkg, layer='catchment_new')
catchment = geemap.geopandas_to_ee(catchment_new)

if show_map:
    Map.addLayer(catchment, {'color': 'darkgrey'}, "Catchment")
    Map.centerObject(catchment, zoom=9)
    display(Map)
```

### Calculate weighted average geopotential and convert to elevation

```{code-cell} ipython3
# execute reducer
dict = image.reduceRegion(ee.Reducer.mean(),
                          geometry=catchment,
                          crs='EPSG:4326',
                          crsTransform=transform)

# get mean value and print
mean_val = dict.getInfo()['z']
ele_dat = mean_val / 9.80665
print(f'geopotential mean: {mean_val:.2f}, elevation: {ele_dat:.2f}m.a.s.l.')
```

***

+++

### ERA5L Temperature and Precipitation Time Series

```{code-cell} ipython3
import pandas as pd
import datetime

def setProperty(image):
    dict = image.reduceRegion(ee.Reducer.mean(), catchment)
    return image.set(dict)
```

```{code-cell} ipython3
collection = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_RAW')\
    .select('temperature_2m','total_precipitation_sum')\
    .filterDate(date_range[0], date_range[1])

withMean = collection.map(setProperty)
```

```{code-cell} ipython3
%%time

df = pd.DataFrame()
df['ts'] = withMean.aggregate_array('system:time_start').getInfo()
df['temp'] = withMean.aggregate_array('temperature_2m').getInfo()
df['temp_c'] = df['temp'] - 273.15
df['prec'] = withMean.aggregate_array('total_precipitation_sum').getInfo()
df['prec'] = df['prec'] * 1000
df['dt'] = df['ts'].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000))
```

```{code-cell} ipython3
df.to_csv(dir_output + 'ERA5L.csv',header=True,index=False)
```

```{code-cell} ipython3
display(df)
```

Append data reference elevation to settings.yml.

```{code-cell} ipython3
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
```

```{code-cell} ipython3
%reset -f
```
