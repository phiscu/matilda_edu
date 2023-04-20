# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Initialization

# +
# Google Earth Engine packages
import ee
import geemap

# other packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Define a function to plot the digital elevation model
def plotFigure(data, label, cmap='Blues'):
    plt.figure(figsize=(12,10))
    plt.imshow(data, extent=grid.extent)
    plt.colorbar(label=label)
    plt.grid()
    

# constants
ee_img = 'Image'
ee_ico = 'ImageCollection'
# -

# initialize GEE at the beginning of session
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()         # authenticate when using GEE for the first time
    ee.Initialize()

# +
import configparser
import ast

# read local config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

# get file config from config.ini
output_folder = config['FILE_SETTINGS']['DIR_OUTPUT']
filename = output_folder + config['FILE_SETTINGS']['DEM_FILENAME']
output_gpkg = output_folder + config['FILE_SETTINGS']['GPKG_NAME']

# get used GEE DEM, coords and other settings
dem = ast.literal_eval(config['CONFIG']['DEM'])
y, x = ast.literal_eval(config['CONFIG']['COORDS'])
show_map = config.getboolean('CONFIG','SHOW_MAP')
# -

# # Start GEE and find catchment area

# Start with base map

if show_map:
    Map = geemap.Map()
    display(Map)
else:
    print("Map view disabled in config.ini")

# Load selected DEM from GEE catalog and add as layer to map

# +
if dem[0] == ee_img:
    image = ee.Image(dem[1])
elif dem[0] == ee_ico:
    image = ee.ImageCollection(dem[1]).select(dem[2]).mosaic()

if show_map:
    srtm_vis = { 'bands': dem[2],
                 'min': 0,
                 'max': 6000,
                'palette': ['000000', '478FCD', '86C58E', 'AFC35E', '8F7131','B78D4F', 'E2B8A6', 'FFFFFF']
               }

    Map.addLayer(image, srtm_vis, dem[3], True, 0.7)
# -

# Add configured discharge point to map and automatically draw box with 30km in all directions

# +
point = ee.Geometry.Point(x,y)
box = point.buffer(30000).bounds()

if show_map:
    Map.addLayer(point,{'color': 'blue'},'Discharge Point')
    Map.addLayer(box,{'color': 'grey'},'Catchment Area', True, 0.7)
    Map.centerObject(box, zoom=9)
# -

# Discharge point (marker) and box (polygon/rectangle) can be added manually to the map above. If features have been drawn, they will overrule the configured discharge point and automatically created box.

if show_map:
    for feature in Map.draw_features:
        f_type = feature.getInfo()['geometry']['type']
        if f_type == 'Point':
            point = feature.geometry()
            print("Manually set pouring point will be considered")
        elif f_type == 'Polygon':
            box = feature.geometry()
            print("Manually drawn box will be considered")

# Export DEM as .tif file to output folder.

geemap.ee_export_image(image, filename=filename, scale=30, region=box, file_per_band=False)

# # Catchment deliniation

# Use <code>pysheds</code> module to determine catchment area for discharge point. The result will be a raster.

# +
# %%time

# GIS packages
from pysheds.grid import Grid
import fiona

# load DEM
DEM_file = filename
grid = Grid.from_raster(DEM_file)
dem = grid.read_raster(DEM_file)

# +
# %%time

# Fill depressions in DEM
flooded_dem = grid.fill_depressions(dem)
# Resolve flats in DEM
inflated_dem = grid.resolve_flats(flooded_dem)

# Specify directional mapping
#N    NE    E    SE    S    SW    W    NW
dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
# Compute flow directions
fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
#catch = grid.catchment(x=x, y=y, fdir=fdir, dirmap=dirmap, xytype='coordinate')
# Compute accumulation
acc = grid.accumulation(fdir)
# Snap pour point to high accumulation cell
x_snap, y_snap = grid.snap_to_mask(acc > 1000, (x, y))
# Delineate the catchment
catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype='coordinate')
# Clip the DEM to the catchment
grid.clip_to(catch)
clipped_catch = grid.view(catch)
# -

demView = grid.view(dem, nodata=np.nan)
plotFigure(demView,'Elevation')
plt.show()

# Convert catchment raster to polygon and save to output folder as geopackage. 

# +
from shapely.geometry import Polygon
import pyproj
from shapely.geometry import shape
from shapely.ops import transform

## Create shapefile and save it
shapes = grid.polygonize()

schema = {
    'geometry': 'Polygon',
    'properties': {'LABEL': 'float:16'}
}

catchment_shape = {}
layer_name = 'catchment_orig'
with fiona.open(output_gpkg, 'w',
                #driver='ESRI Shapefile',#
                driver='GPKG',
                layer=layer_name,
                crs=grid.crs.srs,
                schema=schema) as c:
    i = 0
    for shape, value in shapes:
        catchment_shape = shape
        rec = {}
        rec['geometry'] = shape
        rec['properties'] = {'LABEL' : str(value)}
        rec['id'] = str(i)
        c.write(rec)
        i += 1 

print(f"Layer '{layer_name}' added to GeoPackage '{output_gpkg}'\n")
        
catchment_bounds = [int(np.nanmin(demView)),int(np.nanmax(demView))]
ele_cat = float(np.nanmean(demView))
print(f"Catchment elevation is between {catchment_bounds[0]} m and {catchment_bounds[1]} m")
print(f"Mean catchment elevation is {str(ele_cat)} m")
# -

# Add catchment area to map and calculate area.

# +
catchment = ee.Geometry.Polygon(catchment_shape['coordinates'])
if show_map:
    Map.addLayer(catchment, {}, 'Catchment')

catchment_area = catchment.area().divide(1000*1000).getInfo()
print(f"Catchment area is {catchment_area:.2f} km²")
# -

# # Determine glaciers in catchment area

# Find all glacier that are intersecting catchment area in RGI60 database (for area 13)

# +
import geopandas as gpd

# load catcment and RGI regions as DF
catchment = gpd.read_file(output_gpkg, layer='catchment_orig')
df_areas = gpd.read_file('https://www.glims.org/RGI/rgi60_files/00_rgi60_regions.zip')

# +
import utm
from pyproj import CRS

# get UTM zone for catchment
centroid = catchment.centroid
utm = utm.from_latlon(centroid.y[0],centroid.x[0])
print(f"UTM zone '{utm[2]}', band '{utm[3]}'")

# get CRS based on UTM
crs = CRS.from_dict({'proj':'utm', 'zone':utm[2], 'south':False})

catchment_area = catchment.to_crs(crs).area[0] / 1000 / 1000
print(f"Catchment area (projected) is {catchment_area:.2f} km²")

# +
df_areas = df_areas.set_crs('EPSG:4326',allow_override=True)
catchment = catchment.to_crs('EPSG:4326')
df_areas_catchment = gpd.sjoin(df_areas, catchment, how="inner", predicate="intersects")

if len(df_areas_catchment.index) == 0:
    print('No area found for catchment')
    area = None
elif len(df_areas_catchment.index) == 1:
    area = df_areas_catchment.iloc[0]['RGI_CODE']
    print(f"Catchment belongs to area {area} ({df_areas_catchment.iloc[0]['FULL_NAME']})")
else:
    display(df_areas_catchment)

# +
# %%time

import urllib.request
import re

if area != None:
    url = "https://www.glims.org/RGI/rgi60_files/"  # Replace with the URL of your web server
    html_page = urllib.request.urlopen(url)
    html_content = html_page.read().decode("utf-8")

    # Use regular expressions to find links to files
    pattern = re.compile(r'href="([^"]+\.zip)"')
    file_links = pattern.findall(html_content)


    for file in file_links:
        splits = file.split("_")
        if splits[0] != str(area): 
            continue

        # starting scanning areas
        areaname = splits[0] + " (" + splits[2].split(".")[0] + ")"
        print(f'scanning area {areaname}')

        # read zip into dataframe
        rgi = gpd.read_file(url+file)
        if rgi.crs != catchment.crs:
            print("CRS adjusted")
            catchment = catchment.to_crs(rgi.crs)

        # check whether catchment intersects with glaciers of area
        rgi_catchment = gpd.sjoin(rgi,catchment,how='inner',predicate='intersects')
        if len(rgi_catchment.index) > 0:
            print(f'{len(rgi_catchment.index)} outlines found in area {areaname}\n')
# -

# Some glaciers do not belong to catchment but are intersecting the derived catchment area. Therefore, the percentage of the glacier will be calculated to determine whether glacier will be part of catchment or not (>=50% of its area needs to be in catchment). Glaciers outside catchment with overlapping area will reduce catchment area.
# Results for each glacier can be printed if needed.

# +
# intersects selects too many. calculate percentage of glacier area that is within catchment
rgi_catchment['rgi_area'] = rgi_catchment.to_crs(crs).area    
    
gdf_joined = gpd.overlay(catchment,rgi_catchment, how='union')
gdf_joined['area_joined'] = gdf_joined.to_crs(crs).area
gdf_joined['share_of_area'] = (gdf_joined['area_joined'] / gdf_joined['rgi_area'] * 100)

results = (gdf_joined
           .groupby(['RGIId','LABEL_1'])
           .agg({'share_of_area':'sum'}))

#print(results.sort_values(['share_of_area'],ascending=False))

# +
rgi_catchment = pd.merge(rgi_catchment, results, on="RGIId")
rgi_in_catchment = rgi_catchment.loc[rgi_catchment['share_of_area'] >= 50]
rgi_out_catchment = rgi_catchment.loc[rgi_catchment['share_of_area'] < 50]

catchment_new = gpd.overlay(catchment, rgi_out_catchment, how='difference')
catchment_new = gpd.overlay(catchment_new, rgi_in_catchment, how='union')
catchment_new = catchment_new.dissolve()[['LABEL_1','geometry']]

# +
#catchment_new['area'] = catchment_new.to_crs("+proj=cea +lat_0=35.68250088833567 +lon_0=139.7671 +units=m")['geometry'].area
#area_glac = rgi_in_catchment.to_crs("+proj=cea +lat_0=35.68250088833567 +lon_0=139.7671 +units=m")['geometry'].area
catchment_new['area'] = catchment_new.to_crs(crs)['geometry'].area
area_glac = rgi_in_catchment.to_crs(crs)['geometry'].area

area_glac = area_glac.sum()/1000000
area_cat = catchment_new.iloc[0]['area']/1000000
lat = catchment_new.centroid.to_crs('EPSG:4326').y[0]
print(f"New catchment area is {area_cat} km²")
print(f"Glacierized catchment area is {area_glac} km²")
# -

# Export data to existing geopackage:
# <ul>
#     <li>RGI glaciers within catchment</li>
#     <li>RGI glaciers outside catchment</li>
#     <li>Adjusted catchment area based in RGI glaciers</li>
# </ul>

# +
rgi_in_catchment.to_file(output_gpkg, layer='rgi_in', driver='GPKG')
print(f"Layer 'rgi_in' added to GeoPackage '{output_gpkg}'")

rgi_out_catchment.to_file(output_gpkg, layer='rgi_out', driver='GPKG')
print(f"Layer 'rgi_out' added to GeoPackage '{output_gpkg}'")

catchment_new.to_file(output_gpkg, layer='catchment_new', driver='GPKG')
print(f"Layer 'catchment_new' added to GeoPackage '{output_gpkg}'")
# -

# Add determined glaciers and new catchment area to map.

# +
c_new = geemap.geopandas_to_ee(catchment_new)
rgi = geemap.geopandas_to_ee(rgi_in_catchment)

if show_map:
    Map.addLayer(c_new, {'color': 'orange'}, "Catchment New")
    Map.addLayer(rgi, {'color': 'white'}, "RGI60")
# -

ele_cat = image.reduceRegion(ee.Reducer.mean(),
                          geometry=c_new).getInfo()['dem'] #['elevation']
print(f"Mean catchment elevation (adjusted) is {str(ele_cat)} m")


# The thickness of each glacier must be determined from raster files. Depending on the RGI IDs that are within catchment area, the corresponding raster files will be downloaded from server and stored in output folder.
# The thinkness raster files will be supported by DEM raster files for easier processing. 

# +
def getArchiveNames(row):
    split = row['RGIId'].split('-')
    area = split[1].split('.')[0]
    id = int(split[1].split('.')[1]) // 1000 + 1
    return f'ice_thickness_RGI60-{area}_{id}', f'dem_surface_DEM_RGI60-{area}_{id}'


# determine relevant .zip files for derived RGI IDs 
df_rgiids = pd.DataFrame(rgi_in_catchment['RGIId'].sort_values())
df_rgiids[['thickness', 'dem']] = df_rgiids.apply(getArchiveNames, axis=1, result_type='expand')
zips_thickness = df_rgiids['thickness'].drop_duplicates()
zips_dem = df_rgiids['dem'].drop_duplicates()

# +
from resourcespace import ResourceSpace

# use guest credentials to access media server
api_base_url = 'https://rs.cms.hu-berlin.de/matilda/api/?'  
private_key = '9a19c0cee1cde5fe9180c31c27a8145bc6f7a110cfaa3806ba262eb63d16f086' 
user = 'gast' 

myrepository = ResourceSpace(api_base_url, user, private_key)

# get resource IDs for each .zip file
refs_thickness = pd.DataFrame(myrepository.get_collection_resources(12))[['ref', 'file_size', 'file_extension', 'field8']]
refs_dem = pd.DataFrame(myrepository.get_collection_resources(21))[['ref', 'file_size', 'file_extension', 'field8']]
# -

# reduce list of resources two required zip files 
refs_thickness = pd.merge(zips_thickness, refs_thickness, left_on='thickness', right_on='field8')
refs_dem = pd.merge(zips_dem, refs_dem, left_on='dem', right_on='field8')

# # Retrieve raster files for thickness and corresponding DEM raster files

# **Ice thickness**: download relevant archives from server and extract `.tif` files

# +
# %%time

import requests
from zipfile import ZipFile
import io

cnt_thickness = 0
file_names_thickness = []
for idx,row in refs_thickness.iterrows():   
    content = myrepository.get_resource_file(row['ref'])    
    with ZipFile(io.BytesIO(content), 'r') as zipObj:
        # Get a list of all archived file names from the zip
        listOfFileNames = zipObj.namelist()
        for rgiid in df_rgiids.loc[df_rgiids['thickness']==row['field8']]['RGIId']:
            filename = rgiid + '_thickness.tif'
            if filename in listOfFileNames:
                cnt_thickness += 1
                zipObj.extract(filename, output_folder+'RGI')
                file_names_thickness.append(filename)
            else:
                print(f'File not found: {filename}')
                
print(f'{cnt_thickness} files have been extracted (ice thickness)')
# -

# **DEM**: download relevant archives from server and extract `.tif` files

# +
# %%time

cnt_dem = 0
file_names_dem = []
for idx,row in refs_dem.iterrows():   
    content = myrepository.get_resource_file(row['ref'])    
    with ZipFile(io.BytesIO(content), 'r') as zipObj:
        # Get a list of all archived file names from the zip
        listOfFileNames = zipObj.namelist()
        for rgiid in df_rgiids.loc[df_rgiids['dem']==row['field8']]['RGIId']:
            filename = f"surface_DEM_{rgiid}.tif"
            if filename in listOfFileNames:
                cnt_dem += 1
                zipObj.extract(filename, output_folder+'RGI')
                file_names_dem.append(filename)
            else:
                print(f'File not found: {filename}')
                
print(f'{cnt_dem} files have been extracted (DEM)')
# -

# # Glacier profile creation
# Overlay ice thickness and DEM tif for each glacier to create tuples

# +
from osgeo import gdal

df_all = pd.DataFrame()
if cnt_thickness != cnt_dem:
    print('Number of thickness raster files does not match number of DEM raster files!')
else:
    for idx,rgiid in enumerate(df_rgiids['RGIId']):
        if rgiid in file_names_thickness[idx] and rgiid in file_names_dem[idx]:
            file_list = [
                output_folder + 'RGI/' + file_names_thickness[idx],
                output_folder + 'RGI/' + file_names_dem[idx]
            ]
            array_list = []

            # Read arrays
            for file in file_list:
                src = gdal.Open(file)
                geotransform = src.GetGeoTransform() # Could be done more elegantly outside the for loop
                projection = src.GetProjectionRef()
                array_list.append(src.ReadAsArray())
                pixelSizeX = geotransform[1]
                pixelSizeY =-geotransform[5]                
                src = None
            
            df = pd.DataFrame()
            df['thickness'] = array_list[0].flatten()
            df['altitude'] = array_list[1].flatten()
            df_all = pd.concat([df_all, df])
        else:
            print(f'Raster files do not match for {rgiid}')
# -

# Remove all points with zero ice thickness and aggregate all points to 10m elevation zones.
#
# Export result as CSV file

if len(df_all) > 0:
    df_all = df_all.loc[df_all['thickness'] > 0]
    df_all.sort_values(by=['altitude'],inplace=True)
    
    # get min/max altitude considering catchment and all glaciers
    alt_min = 10*int(min(catchment_bounds[0],df_all['altitude'].min())/10)
    alt_max = max(catchment_bounds[1],df_all['altitude'].max())+10
        
    # create bins in 10m steps
    bins = np.arange(alt_min, df_all['altitude'].max()+10, 10)
    
    # aggregate per bin and do some math
    df_agg = df_all.groupby(pd.cut(df_all['altitude'], bins))['thickness'].agg(count='size', mean='mean').reset_index()
    df_agg['Elevation'] = df_agg['altitude'].apply(lambda x: x.left)
    df_agg['Area'] = df_agg['count']*pixelSizeX*pixelSizeY / catchment_new.iloc[0]['area']
    df_agg['WE'] = df_agg['mean']*0.908*1000
    df_agg['EleZone'] = df_agg['Elevation'].apply(lambda x: 100*int(x/100))
    
    # delete empty elevation bands but keep at least one entry per elevation zone
    df_agg=pd.concat([df_agg.loc[df_agg['count']>0],
                      df_agg.loc[df_agg['count']==0].drop_duplicates(['EleZone'],keep='first')]
                    ).sort_index()
    
    df_agg.drop(['altitude', 'count', 'mean'], axis=1, inplace=True)
    df_agg = df_agg.replace(np.nan, 0)
    df_agg.to_csv(output_folder + 'glacier_profile.csv', header=True, index=False)
    print('Glacier profile for catchment successfully created!')

ele_glac = round(df_all.altitude.mean(), 2)
print(f'Average glacier elevation in the catchment: {ele_glac} m.a.s.l.')

# Create a settings.yaml and store the relevant catchment information.

# +
import yaml

settings = {'area_cat': float(area_cat),
            'ele_cat': float(ele_cat),
            'area_glac': float(area_glac),
            'ele_glac': float(ele_glac),
            'lat': float(lat)
           }
with open(output_folder + 'settings.yml', 'w') as f:
    yaml.safe_dump(settings, f)
# -

# %reset -f


