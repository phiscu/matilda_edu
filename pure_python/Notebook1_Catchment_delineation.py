# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Catchment delineation

# %% [markdown]
# We start our workflow by downloading all the static data we need. In this notebook we will...
#
# 1. ...download a **Digital Elevation Model** (DEM) for hydrologic applications,
#
# 2. ...**delineate the catchment** and determine the **catchment area** using your reference position (e.g. the location of your gauging station) as the "pouring point",
#
# 3. ...identify all glaciers within the catchment and download the **glacier outlines and ice thicknesses**,
#
# 4. ...create a **glacier mass profile** based on elevation zones.
#

# %% [markdown]
# First of all, we will read some settings from the `config.ini` file:
#
# - **cloud project** name for the GEE access
# - **input/output** folders for data imports and downloads
# - **filenames** (DEM, GeoPackage)
# - **coordinates** of the defined "pouring" point (Lat/Long)
# - chosen **DEM** from GEE data catalog
# - **show/hide GEE map** in notebooks

# %%
import os
import pandas as pd
import numpy as np
import configparser
import ast
import matplotlib.pyplot as plt
import scienceplots
import warnings
warnings.filterwarnings("ignore", category=UserWarning)     # Suppress Deprecation Warnings

# read local config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

# get file config from config.ini
cloud_project = config['CONFIG']['CLOUD_PROJECT']
output_folder = config['FILE_SETTINGS']['DIR_OUTPUT']
figures_folder = config['FILE_SETTINGS']['DIR_FIGURES']
filename = output_folder + config['FILE_SETTINGS']['DEM_FILENAME']
output_gpkg = output_folder + config['FILE_SETTINGS']['GPKG_NAME']
zip_output = config['CONFIG']['ZIP_OUTPUT']

# create folder for output figures
os.makedirs(figures_folder, exist_ok=True)

# get used GEE DEM, coords and other settings
dem_config = ast.literal_eval(config['CONFIG']['DEM'])
y, x = ast.literal_eval(config['CONFIG']['COORDS'])
show_map = config.getboolean('CONFIG', 'SHOW_MAP')

# get style for matplotlib plots
plt_style = ast.literal_eval(config['CONFIG']['PLOT_STYLE'])
plt.style.use(plt_style)

# print config data
print(f'Google Cloud Project : {cloud_project}')
print(f'DEM to download: {dem_config[3]}')
print(f'Coordinates of discharge point: Lat {y}, Lon {x}')

# %% [markdown]
# Now, the Google Earth Engine (GEE) access can be initialized. If this is the first time you run the notebook on this machine, you need to authenticate. When using <code>mybinder.org</code> you need to authenticate every time a new session has been launched. Follow the instructions on screen or see the guide in &rarr; [Notebook 0](Notebook0_Introduction.ipynb#Authorize-access-for-Google-Earth-Engine).
#
# <div class="alert alert-block alert-info">
#     <b>Note:</b> In order to make this work, the default cloud project in the <code>config.ini</code> 
#     needs to be changed to your own. See the instructions in 
#     <a href="Notebook0_Introduction.ipynb#signing-up-for-google-earth-engine-gee">Notebook 0</a>, step 5
#     for details.
# </div>
#

# %%
import ee

# initialize GEE at the beginning of session
try:
    ee.Initialize(project=cloud_project)
except Exception as e:
    ee.Authenticate()  # authenticate when using GEE for the first time
    ee.Initialize(project=cloud_project)

print("Google Earth Engine Access initialized!")

# %% [markdown]
# ## Start GEE and download DEM

# %% [markdown]
# Once we are set up, we can start working with the data. Let's start with the **base map**, if enabled in `config.ini`. The map can be used to follow the steps as more layers are added throughout the notebook.

# %%
import geemap

if show_map:
    Map = geemap.Map()
    display(Map)
else:
    print("Map view disabled in config.ini")

# %% [markdown]
# Now we can download the DEM from the GEE catalog and add it as a new layer to the map. The default is the [MERIT DEM] (https://developers.google.com/earth-engine/datasets/catalog/MERIT_DEM_v1_0_3), but you can use any DEM available in the *Google Earth Engine Data Catalog* (https://developers.google.com/earth-engine/datasets/catalog) by specifying it in the `config.ini` file.
#

# %%
if dem_config[0] == 'Image':
    image = ee.Image(dem_config[1]).select(dem_config[2])
elif dem_config[0] == 'ImageCollection':
    image = ee.ImageCollection(dem_config[1]).select(dem_config[2]).mosaic()

if show_map:
    srtm_vis = {'bands': dem_config[2],
                'min': 0,
                'max': 6000,
                'palette': ['000000', '478FCD', '86C58E', 'AFC35E', '8F7131', 'B78D4F', 'E2B8A6', 'FFFFFF']
                }

    Map.addLayer(image, srtm_vis, dem_config[3], True, 0.7)

# %% [markdown]
# Next, we add the location of our discharge observations to the map and generate a **40km** buffer box. 
#
# <div class="alert alert-block alert-info">
#     <b>Note:</b> Please check that the default box covers your research area. Alternatively, you can manually adjust the box by drawing a polygon using the tools in the sidebar. <b>If the selected box is too small, the catchment area will be cropped.</b></div>

# %%
point = ee.Geometry.Point(x, y)
box = point.buffer(40000).bounds()

if show_map:
    Map.addLayer(point, {'color': 'blue'}, 'Discharge Point')
    Map.addLayer(box, {'color': 'grey'}, 'Catchment Area', True, 0.7)
    Map.centerObject(box, zoom=9)

# %% [markdown]
# The gauging location (marker) and the box (polygon/rectangle) can also be added manually to the map above. If features have been drawn, they will overrule the configured discharge point and automatically created box.
#
# <a id="rp01">**Restart Point #1**</a>

# %%
if show_map:
    for feature in Map.draw_features:
        f_type = feature.getInfo()['geometry']['type']
        if f_type == 'Point':
            point = feature.geometry()
            print("Manually set pouring point will be considered")
        elif f_type == 'Polygon':
            box = feature.geometry()
            print("Manually drawn box will be considered")

# %% [markdown]
# Now we can export the DEM as a `.tif` file for the selected extent to the output folder. Depending on the size of the selected area, this might take a while for processing and downloading.

# %%
import xarray as xr

download_xr = config.getboolean('CONFIG', 'GEE_DOWNLOAD_XR')

if download_xr:
    # new method using Xarray (supports larger areas)
    try:
        print('Get GEE data as Xarray...')
        ic = ee.ImageCollection(image)
        ds = xr.open_dataset(
            ic,
            engine='ee',
            projection=ic.first().select(0).projection(),
            geometry=box
        )

        print('Prepare Xarray for GeoTiff conversion...')
        ds_t = ds.isel(time=0).drop_vars("time").transpose()
        ds_t.rio.set_spatial_dims("lon", "lat", inplace=True)

        print('Save DEM as GeoTiff...')
        ds_t.rio.to_raster(filename)
        print('DEM successfully saved at', filename)
    except:
        print('Error during Xarray routine. Try direct download from GEE...')
        geemap.ee_export_image(image, filename=filename, scale=30, region=box, file_per_band=False)
else:
    # old method using GEE API to download .tif directly
    geemap.ee_export_image(image, filename=filename, scale=30, region=box, file_per_band=False)

# %% [markdown]
# ## Catchment deliniation

# %% [markdown]
# Based on the downloaded DEM file, we can delineate the watershed using the <code>pysheds</code> library. The result will be a raster and displayed at the end of this section.
#
# The full documentation of the <code>pysheds</code> module can be found [here](https://mattbartos.com/pysheds/).
#
# <div class="alert alert-block alert-info">
# <b>Note:</b> The catchment delineation involves several steps with large array operations and can take a moment.</div>

# %%
# %%time

# GIS packages
from pysheds.grid import Grid
import fiona

# load DEM
DEM_file = filename
grid = Grid.from_raster(DEM_file)
dem = grid.read_raster(DEM_file)
print("DEM loaded.")

# %%
# %%time

# Fill depressions in DEM
print("Fill depressions in DEM...")
flooded_dem = grid.fill_depressions(dem)
# Resolve flats in DEM
print("Resolve flats in DEM...")
inflated_dem = grid.resolve_flats(flooded_dem)

# Specify directional mapping
# N    NE    E    SE    S    SW    W    NW
dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
# Compute flow directions
print("Compute flow directions...")
fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
# catch = grid.catchment(x=x, y=y, fdir=fdir, dirmap=dirmap, xytype='coordinate')
# Compute accumulation
print("Compute accumulation...")
acc = grid.accumulation(fdir)
# Snap pour point to high accumulation cell
x_snap, y_snap = grid.snap_to_mask(acc > 1000, (x, y))
# Delineate the catchment
print("Delineate the catchment...")
catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype='coordinate')
# Clip the DEM to the catchment
print("Clip the DEM to the catchment...")
grid.clip_to(catch)
clipped_catch = grid.view(catch)
print("Processing completed.")


# %% [markdown]
# Now let's have a look at the catchment area.

# %%
# Define a function to plot the digital elevation model
def plotFigure(data, label, cmap='Blues'):
    plt.figure(figsize=(12, 10))
    plt.imshow(data, extent=grid.extent, cmap=cmap)
    plt.colorbar(label=label)
    plt.grid()


demView = grid.view(dem, nodata=np.nan)
plotFigure(demView, 'Elevation in Meters', cmap='terrain')
plt.savefig(figures_folder + 'NB1_DEM_Catchment.png')
plt.show()

# %% [markdown]
# For the following steps, we need the catchment outline in polygon form. Thus, we will **convert the raster to a polygon** and save both to the output folder in a **geopackage**. We can calculate the important **catchment statistics** needed for the glacio-hydrological model in Notebook 4 from these files.

# %%
from shapely.geometry import Polygon, shape
from shapely.ops import transform
import pyproj

# Create shapefile and save it
shapes = grid.polygonize()

schema = {
    'geometry': 'Polygon',
    'properties': {'LABEL': 'float:16'}
}

catchment_shape = {}
layer_name = 'catchment_orig'
with fiona.open(output_gpkg, 'w',
                # driver='ESRI Shapefile',#
                driver='GPKG',
                layer=layer_name,
                crs=grid.crs.srs,
                schema=schema) as c:
    i = 0
    for shape, value in shapes:
        catchment_shape = shape
        rec = {}
        rec['geometry'] = shape
        rec['properties'] = {'LABEL': str(value)}
        rec['id'] = str(i)
        c.write(rec)
        i += 1

print(f"Layer '{layer_name}' added to GeoPackage '{output_gpkg}'\n")

catchment_bounds = [int(np.nanmin(demView)), int(np.nanmax(demView))]
ele_cat = float(np.nanmean(demView))
print(f"Catchment elevation ranges from {catchment_bounds[0]} m to {catchment_bounds[1]} m.a.s.l.")
print(f"Mean catchment elevation is {ele_cat:.2f} m.a.s.l.")

# %% [markdown]
# We can also add the catchment polygon to the interactive map. This sends it to GEE and allows us to use a GEE function to calculate its area. Please scroll up to see the results on the map.

# %%
catchment = ee.Geometry.Polygon(catchment_shape['coordinates'])
if show_map:
    Map.addLayer(catchment, {}, 'Catchment')

catchment_area = catchment.area().divide(1000 * 1000).getInfo()
print(f"Catchment area is {catchment_area:.2f} km²")

# %% [markdown]
# <div class="alert alert-block alert-warning">
# <b>Note:</b>
#  Please make sure to leave some buffer between the catchment outline and the applied bounding box (&rarr; <a href='#map'>Jump to map</a>). If you run into problems, please extent the box and repeat the DEM download and catchment delineation (&rarr; use <a href='#rp01'>Restart Point #1</a>).</div>
#
# Example:
#
# 1. The automatically created box for the pouring point (in gray) is not sufficient to cover the entire catchment area &rarr; cropped at the eastern edge.
# 2. Manually drawn box (in blue) has been added to ensure that the catchment is not cropped &rarr; buffer remains on all edges
#
# ![Example for Cropped Catchment](images/gee_catchment_extent.png)
#
# ___

# %% [markdown]
# ## Determine glaciers in catchment area

# %% [markdown]
# To acquire outlines of all glaciers in the catchment we will use the Randolph Glacier Inventory version 6 (RGI 6.0). *While RGI version 7 has been released, there are no fully compatible ice thickness datasets yet.*
#
# > The *Randolph Glacier Inventory* is a global inventory of glacier outlines. It is supplemental to the Global Land Ice Measurements from Space initiative (GLIMS). Production of the RGI was motivated by the Fifth Assessment Report of the Intergovernmental Panel on Climate Change (IPCC AR5).
# >
# > Source: https://www.glims.org/RGI/

# %% [markdown]
# The RGI dataset is divided into 19 so called *first-order regions*.
#
# > RGI regions were developed under only three constraints: that they should resemble commonly recognized glacier domains, that together they should contain all of the world’s glaciers, and that their boundaries should be simple and readily recognizable on a map of the world.
# >
# > Source: [Pfeffer et.al. 2014](https://doi.org/10.3189/2014jog13j176)
#
# ![Map of the RGI regions; the red dots indicate the glacier locations and the blue circles the location of the 254 reference WGMS glaciers used by the OGGM calibration](https://docs.oggm.org/en/v1.2.0/_images/wgms_rgi_map.png)

# %% [markdown]
# In the first step, the RGI region of the catchment area must be determined to access the correct repository. Therefore, the RGI region outlines will be downloaded and joined with the catchment outline.
#
# > Source: [RGI Consortium (2017)](https://doi.org/10.7265/4m1f-gd79)

# %% tags=["output_scroll"]
import geopandas as gpd

# load catchment and RGI regions as DF
catchment = gpd.read_file(output_gpkg, layer='catchment_orig')
df_regions = gpd.read_file('https://www.gtn-g.ch/database/GlacReg_2017.zip', layer='GTN-G_glacier_regions_201707')
display(df_regions)

# %% [markdown]
# For spatial calculations it is crucial to use the correct projection. To avoid inaccuracies due to unit conversions we will project the data to UTM whenever we calculate spatial statistics. The relevant UTM zone and band for the catchment area are determined from the coordinates of the pouring point.

# %%
import utm
from pyproj import CRS

utm_zone = utm.from_latlon(y, x)
print(f"UTM zone '{utm_zone[2]}', band '{utm_zone[3]}'")

# get CRS based on UTM
crs = CRS.from_dict({'proj': 'utm', 'zone': utm_zone[2], 'south': False})

catchment_area = catchment.to_crs(crs).area[0] / 1000 / 1000
print(f"Catchment area (projected) is {catchment_area:.2f} km²")

# %% [markdown]
# Now we can perform a spatial join between the catchment outline and the RGI regions. If the catchment contains any glaciers, the corresponding RGI region is determined in this step.

# %%
df_regions = df_regions.set_crs('EPSG:4326', allow_override=True)
catchment = catchment.to_crs('EPSG:4326')
df_regions_catchment = gpd.sjoin(df_regions, catchment, how="inner", predicate="intersects")

if len(df_regions_catchment.index) == 0:
    print('No area found for catchment')
    rgi_region = None
elif len(df_regions_catchment.index) == 1:
    rgi_region = df_regions_catchment.iloc[0]['RGI_CODE']
    print(f"Catchment belongs to RGI region {rgi_region} ({df_regions_catchment.iloc[0]['FULL_NAME']})")
else:
    print("Catchment belongs to more than one region. This use case is not yet supported.")
    display(df_regions_catchment)
    rgi_region = None
rgi_code = int(df_regions_catchment['RGI_CODE'].iloc[0])

# %% [markdown]
# In the next step, the glacier outlines for the determined RGI region will be downloaded. First, we access the repository...

# %%
from resourcespace import ResourceSpace
# use guest credentials to access media server
api_base_url = config['MEDIA_SERVER']['api_base_url']
private_key = config['MEDIA_SERVER']['private_key']
user = config['MEDIA_SERVER']['user']

myrepository = ResourceSpace(api_base_url, user, private_key)

print("Accessed remote repository")

# get resource IDs for each .zip file
rgi_refs = pd.DataFrame(myrepository.get_collection_resources(1168))[
    ['ref', 'file_size', 'file_extension', 'field8']]

if not rgi_refs.empty:
    print("Listing files ...")
    display(rgi_refs)
else:
    print(f'No files found. Please check remote repository.')

# %% [markdown]
# ...and download the `.shp` files for the target region.

# %%
# %%time

import os
import io
from zipfile import ZipFile

output_dir = os.path.join(output_folder, 'RGI')
os.makedirs(output_dir, exist_ok=True)

cnt_rgi = 0
file_names_rgi = []

region_code_str = f'rgi60_{rgi_code:02d}'
#filtering the .zip archives to match our catchment area
filtered_refs = rgi_refs[rgi_refs['field8'] == region_code_str]

if filtered_refs.empty:
    print(f'No RGI archive found for region {rgi_code}')
else:
    print(f'Found RGI archive(s) for region {rgi_code}:')
    display(filtered_refs)

# extracting files from the .zip archives
    for _, row in filtered_refs.iterrows():
        content = myrepository.get_resource_file(row['ref'], row['file_extension'])
        with ZipFile(io.BytesIO(content), 'r') as zipObj:
            zipObj.extractall(output_dir)
            extracted = zipObj.namelist()
            file_names_rgi.extend(extracted)
            cnt_rgi += len(extracted)

    print(f'{cnt_rgi} files extracted to: {output_dir}')

#reading the shapefile
import glob

region_str = f"{rgi_code}_rgi60_*.shp"
search_pattern = os.path.join(output_folder, 'RGI', region_str)

matching_files = glob.glob(search_pattern)

if matching_files:
    rgi_path = matching_files[0]
    rgi = gpd.read_file(rgi_path)
    print(f"Loaded: {rgi_path}")
else:
    print(f"no shapefile for {rgi_code} found")


# %% [markdown]
# Now we can perform a spatial join to determine all glacier outlines that intersect with the catchment area.

# %%
if rgi.crs != catchment.crs:
    print("CRS adjusted")
    catchment = catchment.to_crs(rgi.crs)

# check whether catchment intersects with glaciers of region
print('Perform spatial join...')
rgi_catchment = gpd.sjoin(rgi, catchment, how='inner', predicate='intersects')
if len(rgi_catchment.index) > 0:
    print(f'{len(rgi_catchment.index)} outlines loaded from RGI Region {rgi_code}\n')

# %% [markdown]
# Some glaciers are not actually in the catchment, but intersect its outline due to spatial inaccuracies. We will first determine their fractional overlap with the target catchment.

# %% tags=["output_scroll"]
# intersects selects too many. calculate percentage of glacier area that is within catchment
rgi_catchment['rgi_area'] = rgi_catchment.to_crs(crs).area

gdf_joined = gpd.overlay(catchment, rgi_catchment, how='union')
gdf_joined['area_joined'] = gdf_joined.to_crs(crs).area
gdf_joined['share_of_area'] = round((gdf_joined['area_joined'] / gdf_joined['rgi_area'] * 100), 2)

results = (gdf_joined
           .groupby(['RGIId', 'LABEL_1'])
           .agg({'share_of_area': 'sum'}))

display(results.sort_values(['share_of_area'], ascending=False))

# %% [markdown]
# Now we can **filter** based on the percentage of shared area. After that the catchment area will be adjusted as follows:
#
# - **&#8805;50%** of the area are in the catchment &rarr; **include** and extend catchment area by full glacier outlines (if needed)
# - **<50%** of the area are in the catchment &rarr; **exclude** and reduce catchment area by glacier outlines (if needed)

# %%
rgi_catchment_merge = pd.merge(rgi_catchment, results, on="RGIId")
rgi_in_catchment = rgi_catchment_merge.loc[rgi_catchment_merge['share_of_area'] >= 50]
rgi_out_catchment = rgi_catchment_merge.loc[rgi_catchment_merge['share_of_area'] < 50]
catchment_new = gpd.overlay(catchment, rgi_out_catchment, how='difference')
catchment_new = gpd.overlay(catchment_new, rgi_in_catchment, how='union')
catchment_new = catchment_new.dissolve()[['LABEL_1', 'geometry']]

print(f'Total number of determined glacier outlines: {len(rgi_catchment_merge)}')
print(f'Number of included glacier outlines (overlap >= 50%): {len(rgi_in_catchment)}')
print(f'Number of excluded glacier outlines (overlap < 50%): {len(rgi_out_catchment)}')

# %% [markdown]
# The RGI-IDs of the remaining glaciers are stored in `Glaciers_in_catchment.csv`.

# %% tags=["output_scroll"]
from pathlib import Path

Path(output_folder + 'RGI').mkdir(parents=True, exist_ok=True)

glacier_ids = pd.DataFrame(rgi_in_catchment)
glacier_ids['RGIId'] = glacier_ids['RGIId'].map(lambda x: str(x).lstrip('RGI60-'))
glacier_ids.to_csv(output_folder + 'RGI/' + 'Glaciers_in_catchment.csv', columns=['RGIId', 'GLIMSId'], index=False)
display(glacier_ids)

# %% [markdown]
# With the updated catchment outline we can now determine the **final area of the catchment** and the **part covered by glaciers**.

# %%
catchment_new['area'] = catchment_new.to_crs(crs)['geometry'].area
area_glac = rgi_in_catchment.to_crs(crs)['geometry'].area

area_glac = area_glac.sum() / 1000000
area_cat = catchment_new.iloc[0]['area'] / 1000000
cat_cent = catchment_new.to_crs(crs).centroid
lat = cat_cent.to_crs('EPSG:4326').y[0]

print(f"New catchment area is {area_cat:.2f} km²")
print(f"Glacierized catchment area is {area_glac:.2f} km²")

# %% [markdown]
# The files just created are added to the existing geopackage...

# %%
rgi_in_catchment.to_file(output_gpkg, layer='rgi_in', driver='GPKG')
print(f"Layer 'rgi_in' added to GeoPackage '{output_gpkg}'")

rgi_out_catchment.to_file(output_gpkg, layer='rgi_out', driver='GPKG')
print(f"Layer 'rgi_out' added to GeoPackage '{output_gpkg}'")

catchment_new.to_file(output_gpkg, layer='catchment_new', driver='GPKG')
print(f"Layer 'catchment_new' added to GeoPackage '{output_gpkg}'")

# %% [markdown]
# ...and can also be added to the interactive map...

# %%
c_new = geemap.geopandas_to_ee(catchment_new)
rgi = geemap.geopandas_to_ee(rgi_in_catchment)

if show_map:
    Map.addLayer(c_new, {'color': 'orange'}, "Catchment New")
    Map.addLayer(rgi, {'color': 'white'}, "RGI60")
    print('New layers added.')

# %% [markdown]
# ...or combined in a simple plot.

# %%
fig, ax = plt.subplots()
catchment_new.plot(color='tan', ax=ax)
rgi_in_catchment.plot(color="white", edgecolor="black", ax=ax)
plt.scatter(x, y, facecolor='blue', s=100)
plt.title("Catchment Area with Pouring Point and Glaciers")
plt.savefig(figures_folder + 'NB1_Glaciers_Catchment.png')
plt.show()

# %% [markdown]
# After adding the new catchment area to GEE, we can easily calculate the mean catchment elevation in meters above sea level.

# %%
ele_cat = image.reduceRegion(ee.Reducer.mean(),
                             geometry=c_new).getInfo()[dem_config[2]]
print(f"Mean catchment elevation (adjusted) is {ele_cat:.2f} m a.s.l.")


# %% [markdown]
# ### Interim Summary:
#
# So far we have...
#
# - ...delineated the catchment and determined its area,
#
# - ...calculated the average elevation of the catchment,
#
# - ...identified the glaciers in the catchment and calculated their combined area.
#
# In the next step, we will create a glacier profile to determine how the ice is distributed over the elevation range.
# ___

# %% [markdown]
# ## Retrieve ice thickness rasters and corresponding DEM files

# %% [markdown]
# Determining ice thickness from remotely sensed data is a challenging task. Fortunately, [Farinotti et.al. (2019)](https://doi.org/10.1038/s41561-019-0300-3) calculated an ensemble estimate of different methods for all glaciers in RGI6 and made the data available [to the public](https://www.research-collection.ethz.ch/handle/20.500.11850/315707).

# %% [markdown]
# The published repository contains...
#
# > (a) the **ice thickness distribution** of individual glaciers,<br/>
# > (b) global grids at various resolutions with **summary-information about glacier number, area, and volume**, and<br/>
# > (c) the **digital elevation models** of the glacier surfaces used to produce the estimates.
# >
# > Nomenclature for glaciers and regions follows the Randolph Glacier Inventory (RGI) version 6.0.
# >
# > Source: Farinotti et.al. 2019 - [README](https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/315707/README.txt)
#
# The **ice thickness rasters (a)** and **aligned DEMs (c)** are the perfect input data for our glacier profile. The files are selected and downloaded by their **RGI IDs** and stored in the output folder.
#
# Since the original files hosted by ETH Zurich are stored in large archives, we cut the dataset into smaller slices and reupload them according to the respective [license](https://creativecommons.org/licenses/by-nc-sa/4.0/) to make them searchable, improve performance, and limit traffic.
#
# First, we identify the relevant archives for our set of glacier IDs.

# %%
def getArchiveNames(row):
    region = row['RGIId'].split('.')[0]
    id = (int(row['RGIId'].split('.')[1]) - 1) // 1000 + 1
    return f'ice_thickness_{region}_{id}', f'dem_surface_DEM_{region}_{id}'


# determine relevant .zip files for derived RGI IDs
df_rgiids = pd.DataFrame(rgi_in_catchment['RGIId'].sort_values())
df_rgiids[['thickness', 'dem']] = df_rgiids.apply(getArchiveNames, axis=1, result_type='expand')
zips_thickness = df_rgiids['thickness'].drop_duplicates()
zips_dem = df_rgiids['dem'].drop_duplicates()

print(f'Thickness archives:\t{zips_thickness.tolist()}')
print(f'DEM archives:\t\t{zips_dem.tolist()}')

# %% [markdown]
# The archives are stored on a file server at the Humboldt University of Berlin, which provides limited read access to this notebook. The corresponding credentials and API key are defined in the `config.ini` file. The next step is to identify the corresponding resource references for the previously identified archives.

# %%
from resourcespace import ResourceSpace

# use guest credentials to access media server
api_base_url = config['MEDIA_SERVER']['api_base_url']
private_key = config['MEDIA_SERVER']['private_key']
user = config['MEDIA_SERVER']['user']

myrepository = ResourceSpace(api_base_url, user, private_key)

# get resource IDs for each .zip file
refs_thickness = pd.DataFrame(myrepository.get_collection_resources(12))[
    ['ref', 'file_size', 'file_extension', 'field8']]
refs_dem = pd.DataFrame(myrepository.get_collection_resources(21))[['ref', 'file_size', 'file_extension', 'field8']]

# reduce list of resources two required zip files
refs_thickness = pd.merge(zips_thickness, refs_thickness, left_on='thickness', right_on='field8')
refs_dem = pd.merge(zips_dem, refs_dem, left_on='dem', right_on='field8')

print(f'Thickness archive references:\n')
display(refs_thickness)
print(f'DEM archive references:\n')
display(refs_dem)

# %% [markdown]
# Again, depending on the number of files and bandwidth, this may take a moment. Let's start with the **ice thickness**...

# %%
# %%time

import requests
from zipfile import ZipFile
import io

cnt_thickness = 0
file_names_thickness = []
for idx, row in refs_thickness.iterrows():
    content = myrepository.get_resource_file(row['ref'], row['file_extension'])
    with ZipFile(io.BytesIO(content), 'r') as zipObj:
        # Get a list of all archived file names from the zip
        listOfFileNames = zipObj.namelist()
        for rgiid in df_rgiids.loc[df_rgiids['thickness'] == row['field8']]['RGIId']:
            filename = rgiid + '_thickness.tif'
            if filename in listOfFileNames:
                cnt_thickness += 1
                zipObj.extract(filename, output_folder + 'RGI')
                file_names_thickness.append(filename)
            else:
                print(f'File not found: {filename}')

print(f'{cnt_thickness} files have been extracted (ice thickness)')

# %% [markdown]
# ...and continue with the matching **DEMs**.

# %%
# %%time

cnt_dem = 0
file_names_dem = []
for idx, row in refs_dem.iterrows():
    content = myrepository.get_resource_file(row['ref'])
    with ZipFile(io.BytesIO(content), 'r') as zipObj:
        # Get a list of all archived file names from the zip
        listOfFileNames = zipObj.namelist()
        for rgiid in df_rgiids.loc[df_rgiids['dem'] == row['field8']]['RGIId']:
            filename = f"surface_DEM_{rgiid}.tif"
            if filename in listOfFileNames:
                cnt_dem += 1
                zipObj.extract(filename, output_folder + 'RGI')
                file_names_dem.append(filename)
            else:
                print(f'File not found: {filename}')

print(f'{cnt_dem} files have been extracted (DEM)')

# %% [markdown]
# <div class="alert alert-block alert-warning">
# <b>Note:</b>
#  Please check whether all files have been extracted to the output folder without error messages and <b>the number of files matches the number of glaciers</b>.</div>

# %%
if len(rgi_in_catchment) == cnt_thickness == cnt_dem:
    print(f"Number of files matches the number of glaciers within catchment: {len(rgi_in_catchment)}")
else:
    print("There is a mismatch of extracted files. Please check previous steps for error messages!")
    print(f'Number of included glaciers:\t{len(rgi_in_catchment)}')
    print(f'Ice thickness files:\t\t{cnt_thickness}')
    print(f'DEM files:\t\t\t{cnt_dem}')

# %% [markdown]
# ## Glacier profile creation
#
# The **glacier profile** is used to pass the distribution of ice mass in the catchment to the glacio-hydrological model in **Notebook 4**, following the approach of [Seibert et.al.2018](https://doi.org/10.5194/hess-22-2211-2018). The model then calculates the annual mass balance and redistributes the ice mass accordingly.
#
# To derive the profile from spatially distributed data, we first stack the ice thickness and corresponding DEM rasters for each glacier and create tuples of ice thickness and elevation values.

# %%
from osgeo import gdal

df_all = pd.DataFrame()
if cnt_thickness != cnt_dem:
    print('Number of ice thickness raster files does not match number of DEM raster files!')
else:
    for idx, rgiid in enumerate(df_rgiids['RGIId']):
        if rgiid in file_names_thickness[idx] and rgiid in file_names_dem[idx]:
            file_list = [
                output_folder + 'RGI/' + file_names_thickness[idx],
                output_folder + 'RGI/' + file_names_dem[idx]
            ]
            array_list = []

            # Read arrays
            for file in file_list:
                src = gdal.Open(file)
                geotransform = src.GetGeoTransform()  # Could be done more elegantly outside the for loop
                projection = src.GetProjectionRef()
                array_list.append(src.ReadAsArray())
                pixelSizeX = geotransform[1]
                pixelSizeY = -geotransform[5]
                src = None

            df = pd.DataFrame()
            df['thickness'] = array_list[0].flatten()
            df['altitude'] = array_list[1].flatten()
            df_all = pd.concat([df_all, df])

        else:
            print(f'Raster files do not match for {rgiid}')

print("Ice thickness and elevations rasters stacked")
print("Value pairs created")

# %% [markdown]
# Now we can remove all data points with zero ice thickness and aggregate all data points into 10m **elevation zones**. The next step is to calculate the **water equivalent** (WE) from the average ice thickness in each elevation zone.
#
# The result is exported to the output folder as `glacier_profile.csv`.

# %%
if len(df_all) > 0:
    df_all = df_all.loc[df_all['thickness'] > 0]
    df_all.sort_values(by=['altitude'], inplace=True)

    # get min/max altitude considering catchment and all glaciers
    alt_min = 10 * int(min(catchment_bounds[0], df_all['altitude'].min()) / 10)
    alt_max = max(catchment_bounds[1], df_all['altitude'].max()) + 10

    # create bins in 10m steps
    bins = np.arange(alt_min, df_all['altitude'].max() + 10, 10)

    # aggregate per bin and do some math
    df_agg = df_all.groupby(pd.cut(df_all['altitude'], bins))['thickness'].agg(count='size', mean='mean').reset_index()
    df_agg['Elevation'] = df_agg['altitude'].apply(lambda x: x.left).astype(int)
    df_agg['Area'] = df_agg['count'] * pixelSizeX * pixelSizeY / catchment_new.iloc[0]['area']
    df_agg['WE'] = df_agg['mean'] * 0.908 * 1000
    df_agg['EleZone'] = df_agg['Elevation'].apply(lambda x: 100 * int(x / 100))

    # delete empty elevation bands but keep at least one entry per elevation zone
    df_agg = pd.concat([df_agg.loc[df_agg['count'] > 0],
                        df_agg.loc[df_agg['count'] == 0].drop_duplicates(['EleZone'], keep='first')]
                       ).sort_index()

    df_agg.drop(['altitude', 'count', 'mean'], axis=1, inplace=True)
    df_agg = df_agg.replace(np.nan, 0)
    df_agg.to_csv(output_folder + 'glacier_profile.csv', header=True, index=False)
    print('Glacier profile for catchment created!\n')
    display(df_agg)

# %% [markdown]
# Let's visualize the glacier profile. First we aggregate the ice mass in larger elevation zones for better visibility. The level of aggregation can be adjusted using the variable `steps` (default is 20m).

# %%
# aggregation level for plot -> feel free to adjust
steps = 20

# get elevation range where glaciers are present
we_range = df_agg.loc[df_agg['WE'] > 0]['Elevation']
we_range.min() // steps * steps
plt_zones = pd.Series(range(int(we_range.min() // steps * steps),
                            int(we_range.max() // steps * steps + steps),
                            steps), name='EleZone').to_frame().set_index('EleZone')

# calculate glacier mass and aggregate glacier profile to defined elevation steps
plt_data = df_agg.copy()
plt_data['EleZone'] = plt_data['Elevation'].apply(lambda x: int(x // steps * steps))
plt_data['Mass'] = plt_data['Area'] * catchment_new.iloc[0]['area'] * plt_data['WE'] * 1e-9  # mass in Mt
plt_data = plt_data.drop(['Area', 'WE'], axis=1).groupby('EleZone').sum().reset_index().set_index('EleZone')
plt_data = plt_zones.join(plt_data)
display(plt_data)

# %% [markdown]
# Now, we can plot the estimated **glacier mass (in Mt) for each elevation zone.**

# %%
import matplotlib.ticker as ticker

fig, ax = plt.subplots(figsize=(4, 5))
plt_data.plot.barh(y='Mass', ax=ax)
ax.set_xlabel("Glacier mass [Mt]")
ax.set_yticks(ax.get_yticks()[::int(100 / steps)])
ax.set_ylabel("Elevation zone [m a.s.l.]")
ax.get_legend().remove()
plt.title("Initial Ice Distribution")
plt.tight_layout()
plt.savefig(figures_folder + 'NB1_Glacier_Mass_Elevation.png')
plt.show()

# %% [markdown]
# Finally, we calculate the average glacier elevation in meters above sea level.

# %%
ele_glac = round(df_all.altitude.mean(), 2)
print(f'Average glacier elevation in the catchment: {ele_glac:.2f} m a.s.l.')

# %% [markdown]
# ## Store calculated values for other notebooks

# %% [markdown]
# Create a `settings.yml` and store the relevant catchment information for the model setup:
#
# - **area_cat**: area of the catchment in km²
# - **ele_cat**: average elevation of the catchment in m.a.s.l.
# - **area_glac**: glacier covered area as of 2000 in km²
# - **ele_glac**: average elevation of glacier covered area in m.a.s.l.
# - **lat**: latitude of catchment centroid

# %%
import yaml

settings = {'area_cat': float(area_cat),
            'ele_cat': float(ele_cat),
            'area_glac': float(area_glac),
            'ele_glac': float(ele_glac),
            'lat': float(lat)
            }
with open(output_folder + 'settings.yml', 'w') as f:
    yaml.safe_dump(settings, f)

print('Settings saved to file.')
display(pd.DataFrame(settings.items(), columns=['Parameter', 'Value']).set_index('Parameter'))

# %% [markdown]
# You can now continue with [Notebook 2](Notebook2_Forcing_data.ipynb) or ...

# %% [markdown]
# ## *Optional*: Download Outputs
#
# <div class="alert alert-block alert-info">
# <b>Note:</b>
#  The output folder is zipped at the end of each notebook and can be downloaded (file <code>output_download.zip</code>). This is especially useful if you want to use the binder environment again, but don't want to start over from Notebook 1.</div>
#
# <img src="images/download_output.png" width=300>
#
# %%
import shutil

if zip_output:
    shutil.make_archive('output_download', 'zip', 'output')
    print('Output folder can be download now (file output_download.zip)')

# %%
# %reset -f
