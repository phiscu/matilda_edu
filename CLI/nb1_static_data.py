import ee
import pandas as pd
import configparser
import ast
import geemap
from IPython.core.display_functions import display
from pysheds.grid import Grid
import fiona
# import numpy as np
import matplotlib.pyplot as plt
import numpy as np
# from shapely.geometry import Polygon
# import pyproj
# from shapely.geometry import shape
# from shapely.ops import transform
import geopandas as gpd
import utm
from pyproj import CRS
import urllib.request
import re
# import requests
from zipfile import ZipFile
import io
from osgeo import gdal
# import matplotlib.ticker as ticker
import yaml
from pathlib import Path
import sys
import os
# Add change cwd to matilda_edu home dir and add it to PATH
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.append(cwd)            # Hotfix. Master script runs subprocess that changes the CWD.
sys.path.append(parent_dir)

from resourcespace import ResourceSpace

import matplotlib.font_manager as fm
path_to_palatinottf = '/home/phillip/Downloads/Palatino.ttf'
fm.fontManager.addfont(path_to_palatinottf)
plt.rcParams["font.family"] = "Palatino"


## Initialize GEE
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()         # authenticate when using GEE for the first time
    ee.Initialize()

# read local config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

# get file config from config.ini
output_folder = config['FILE_SETTINGS']['DIR_OUTPUT']
filename = output_folder + config['FILE_SETTINGS']['DEM_FILENAME']
output_gpkg = output_folder + config['FILE_SETTINGS']['GPKG_NAME']

# get used GEE DEM, coords and other settings
dem_config = ast.literal_eval(config['CONFIG']['DEM'])
y, x = ast.literal_eval(config['CONFIG']['COORDS'])

# print config data
print(f'Used DEM: {dem_config[3]}')
print(f'Coordinates of discharge point: Lat {y}, Lon {x}')

# add DEM to GEE Image(Collection)
if dem_config[0] == 'Image':
    image = ee.Image(dem_config[1]).select(dem_config[2])
elif dem_config[0] == 'ImageCollection':
    image = ee.ImageCollection(dem_config[1]).select(dem_config[2]).mosaic()

# add pouring point
point = ee.Geometry.Point(x,y)
box = point.buffer(40000).bounds()

# download DEM
geemap.ee_export_image(image, filename=filename, scale=30, region=box, file_per_band=False)

# load DEM from file
DEM_file = filename
grid = Grid.from_raster(DEM_file)
dem = grid.read_raster(DEM_file)
print("DEM loaded.")

# Catchment delineation
# %%time

# Fill depressions in DEM
print("Fill depressions in DEM...")
flooded_dem = grid.fill_depressions(dem)
# Resolve flats in DEM
print("Resolve flats in DEM...")
inflated_dem = grid.resolve_flats(flooded_dem)

# Specify directional mapping
#N    NE    E    SE    S    SW    W    NW
dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
# Compute flow directions
print("Compute flow directions...")
fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
#catch = grid.catchment(x=x, y=y, fdir=fdir, dirmap=dirmap, xytype='coordinate')
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

# Define a function to plot the digital elevation model
def plotFigure(data, label, cmap='Blues'):
    plt.figure(figsize=(12,10))
    plt.imshow(data, extent=grid.extent, cmap=cmap)
    plt.colorbar(label=label)
    plt.grid()

demView = grid.view(dem, nodata=np.nan)
plotFigure(demView,'Elevation in Meters',cmap='terrain')
plt.savefig(output_folder + 'dem_catchment.png')

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

catchment = ee.Geometry.Polygon(catchment_shape['coordinates'])
catchment_area = catchment.area().divide(1000*1000).getInfo()
print(f"Catchment area is {catchment_area:.2f} km²")

# load catcment and RGI regions as DF
catchment = gpd.read_file(output_gpkg, layer='catchment_orig')
df_regions = gpd.read_file('https://www.glims.org/RGI/rgi60_files/00_rgi60_regions.zip')

# Reproject data in UTM
utm_zone = utm.from_latlon(y, x)
print(f"UTM zone '{utm_zone[2]}', band '{utm_zone[3]}'")

# get CRS based on UTM
crs = CRS.from_dict({'proj':'utm', 'zone':utm_zone[2], 'south':False})

catchment_area = catchment.to_crs(crs).area[0] / 1000 / 1000
print(f"Catchment area (projected) is {catchment_area:.2f} km²")

# Determine the correct RGI region using a spatial join

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

# Download all outlines of the detected RGI region and intersect them with the catchment outline
if rgi_region != None:
    url = "https://www.glims.org/RGI/rgi60_files/"  # Replace with the URL of your web server
    html_page = urllib.request.urlopen(url)
    html_content = html_page.read().decode("utf-8")
    print('Reading Randolph Glacier Inventory 6.0 in GLIMS database...')

    # Use regular expressions to find links to files
    pattern = re.compile(r'href="([^"]+\.zip)"')
    file_links = pattern.findall(html_content)


    for file in file_links:
        splits = file.split("_")
        if splits[0] != str(rgi_region):
            continue

        # starting scanning regions
        regionname = splits[0] + " (" + splits[2].split(".")[0] + ")"
        print(f'Locating glacier outlines in RGI Region {regionname}...')

        # read zip into dataframe
        print('Loading shapefiles...')
        rgi = gpd.read_file(url+file)
        if rgi.crs != catchment.crs:
            print("CRS adjusted")
            catchment = catchment.to_crs(rgi.crs)

        # check whether catchment intersects with glaciers of region
        print('Perform spatial join...')
        rgi_catchment = gpd.sjoin(rgi,catchment,how='inner',predicate='intersects')
        if len(rgi_catchment.index) > 0:
            print(f'{len(rgi_catchment.index)} outlines loaded from RGI Region {regionname}\n')

# intersects selects too many. calculate percentage of glacier area that is within catchment
rgi_catchment['rgi_area'] = rgi_catchment.to_crs(crs).area

gdf_joined = gpd.overlay(catchment, rgi_catchment, how='union')
gdf_joined['area_joined'] = gdf_joined.to_crs(crs).area
gdf_joined['share_of_area'] = (gdf_joined['area_joined'] / gdf_joined['rgi_area'] * 100)

results = (gdf_joined
           .groupby(['RGIId', 'LABEL_1'])
           .agg({'share_of_area': 'sum'}))

display(results.sort_values(['share_of_area'], ascending=False))

# filter if less than 50% are in the catchment
rgi_catchment_merge = pd.merge(rgi_catchment, results, on="RGIId")
rgi_in_catchment = rgi_catchment_merge.loc[rgi_catchment_merge['share_of_area'] >= 50]
rgi_out_catchment = rgi_catchment_merge.loc[rgi_catchment_merge['share_of_area'] < 50]
catchment_new = gpd.overlay(catchment, rgi_out_catchment, how='difference')
catchment_new = gpd.overlay(catchment_new, rgi_in_catchment, how='union')
catchment_new = catchment_new.dissolve()[['LABEL_1', 'geometry']]

print(f'Total number of determined glacier outlines: {len(rgi_catchment_merge)}')
print(f'Number of included glacier outlines (overlap >= 50%): {len(rgi_in_catchment)}')
print(f'Number of excluded glacier outlines (overlap < 50%): {len(rgi_out_catchment)}')

# Store RGI-IDs of the remaining glaciers in `Glaciers_in_catchment.csv`.
Path(output_folder + 'RGI').mkdir(parents=True, exist_ok=True)
glacier_ids = pd.DataFrame(rgi_in_catchment)
glacier_ids['RGIId'] = glacier_ids['RGIId'].map(lambda x: str(x).lstrip('RGI60-'))
glacier_ids.to_csv(output_folder + 'RGI/' + 'Glaciers_in_catchment.csv', columns=['RGIId', 'GLIMSId'], index=False)
display(glacier_ids)

# determine the final area of the catchment and the part covered by glaciers.
catchment_new['area'] = catchment_new.to_crs(crs)['geometry'].area
area_glac = rgi_in_catchment.to_crs(crs)['geometry'].area

area_glac = area_glac.sum()/1000000
area_cat = catchment_new.iloc[0]['area']/1000000
cat_cent = catchment_new.to_crs(crs).centroid
lat = cat_cent.to_crs('EPSG:4326').y[0]

print(f"New catchment area is {area_cat:.2f} km²")
print(f"Glacierized catchment area is {area_glac:.2f} km²")

# files added to the existing geopackage
rgi_in_catchment.to_file(output_gpkg, layer='rgi_in', driver='GPKG')
print(f"Layer 'rgi_in' added to GeoPackage '{output_gpkg}'")

rgi_out_catchment.to_file(output_gpkg, layer='rgi_out', driver='GPKG')
print(f"Layer 'rgi_out' added to GeoPackage '{output_gpkg}'")

catchment_new.to_file(output_gpkg, layer='catchment_new', driver='GPKG')
print(f"Layer 'catchment_new' added to GeoPackage '{output_gpkg}'")

# translate adapted catchment and glacier outlines into GEE features
c_new = geemap.geopandas_to_ee(catchment_new)
rgi = geemap.geopandas_to_ee(rgi_in_catchment)

# Plot glaciers in the catchment
fig, ax = plt.subplots()
catchment_new.plot(color='tan',ax=ax)
rgi_in_catchment.plot(color="white",edgecolor="black",ax=ax)
plt.scatter(x, y, facecolor='blue', s=100)
plt.title("Catchment Area with Pouring Point and Glaciers")
plt.savefig(output_folder + 'glaciers_in_catchment.png')

# calculate the mean catchment elevation in meters above sea level.
ele_cat = image.reduceRegion(ee.Reducer.mean(),
                          geometry=c_new).getInfo()[dem_config[2]]
print(f"Mean catchment elevation (adjusted) is {ele_cat:.2f} m a.s.l.")


## Ice thickness estimation
# identify the relevant archives for our set of glacier IDs.
def getArchiveNames(row):
    region = row['RGIId'].split('.')[0]
    id = (int(row['RGIId'].split('.')[1]) - 1) // 1000 + 1
    return f'ice_thickness_RGI60-{region}_{id}', f'dem_surface_DEM_RGI60-{region}_{id}'

# determine relevant .zip files for derived RGI IDs
df_rgiids = pd.DataFrame(rgi_in_catchment['RGIId'].sort_values())
df_rgiids[['thickness', 'dem']] = df_rgiids.apply(getArchiveNames, axis=1, result_type='expand')
zips_thickness = df_rgiids['thickness'].drop_duplicates()
zips_dem = df_rgiids['dem'].drop_duplicates()

print(f'Thickness archives:\t{zips_thickness.tolist()}')
print(f'DEM archives:\t\t{zips_dem.tolist()}')

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

# Download ice thickness tif for all target glaciers
# %%time
cnt_thickness = 0
file_names_thickness = []
for idx, row in refs_thickness.iterrows():
    content = myrepository.get_resource_file(row['ref'])
    with ZipFile(io.BytesIO(content), 'r') as zipObj:
        # Get a list of all archived file names from the zip
        listOfFileNames = zipObj.namelist()
        for rgiid in df_rgiids.loc[df_rgiids['thickness'] == row['field8']]['RGIId']:
            filename = 'RGI60-' + rgiid + '_thickness.tif'
            if filename in listOfFileNames:
                cnt_thickness += 1
                zipObj.extract(filename, output_folder + 'RGI')
                file_names_thickness.append(filename)
            else:
                print(f'File not found: {filename}')

print(f'{cnt_thickness} files have been extracted (ice thickness)')

# Match thickness files with DEMs
# %%time
cnt_dem = 0
file_names_dem = []
for idx, row in refs_dem.iterrows():
    content = myrepository.get_resource_file(row['ref'])
    with ZipFile(io.BytesIO(content), 'r') as zipObj:
        # Get a list of all archived file names from the zip
        listOfFileNames = zipObj.namelist()
        for rgiid in df_rgiids.loc[df_rgiids['dem'] == row['field8']]['RGIId']:
            filename = f"surface_DEM_RGI60-{rgiid}.tif"
            if filename in listOfFileNames:
                cnt_dem += 1
                zipObj.extract(filename, output_folder + 'RGI')
                file_names_dem.append(filename)
            else:
                print(f'File not found: {filename}')

print(f'{cnt_dem} files have been extracted (DEM)')

# Check number of files
if len(rgi_in_catchment) == cnt_thickness == cnt_dem:
    print(f"Number of files matches the number of glaciers within catchment: {len(rgi_in_catchment)}")
else:
    print("There is a mismatch of extracted files. Please check previous steps for error messages!")
    print(f'Number of included glaciers:\t{len(rgi_in_catchment)}')
    print(f'Ice thickness files:\t\t{cnt_thickness}')
    print(f'DEM files:\t\t\t{cnt_dem}')

# stack ice thickness and DEMs and create tuples of ice thickness and elevation values
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

# aggregate into 10m elevation zones and calculate water equivalent (WE). Stored in `glacier_profile.csv`.
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
    df_agg['Elevation'] = df_agg['altitude'].apply(lambda x: x.left)
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


# visualize the glacier profile
steps = 20 # aggregation level for plot -> feel free to adjust

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


fig, ax = plt.subplots(figsize=(4, 5))
plt_data.plot.barh(y='Mass', ax=ax)
ax.set_xlabel("Glacier mass [Mt]")
ax.set_yticks(ax.get_yticks()[::int(100 / steps)])
ax.set_ylabel("Elevation zone [m a.s.l.]")
ax.get_legend().remove()
plt.title("Initial Ice Distribution")
plt.tight_layout()
plt.savefig(output_folder + 'glacier_profile.png')

# Calculate average glacier elevation in meters above sea level.
ele_glac = round(df_all.altitude.mean(), 2)
print(f'Average glacier elevation in the catchment: {ele_glac:.2f} m a.s.l.')

# Store the following values for other notebook in `settings.yml`.

# - area_cat: area of catchment in km²
# - ele_cat: average elevation of catchment in m.a.s.l.
# - area_glac: area of glacier in km²
# - ele_glac: average elevation of glaciers in m.a.s.l.
# - lat: latitude of catchment centroid

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