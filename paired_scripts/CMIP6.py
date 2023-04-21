# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Google Earth Engine packages
import ee
import geemap
import numpy as np

# %%
# initialize GEE at the beginning of session
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()         # authenticate when using GEE for the first time
    ee.Initialize()

# %%
import configparser
import ast

# read local config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

# get file config from config.ini
dir_output = config['FILE_SETTINGS']['DIR_OUTPUT']
output_gpkg = dir_output + config['FILE_SETTINGS']['GPKG_NAME']

# get date range for forcing data
#date_range = ast.literal_eval(config['CONFIG']['DATE_RANGE'])

# %%
import geopandas as gpd

catchment_new = gpd.read_file(output_gpkg, layer='catchment_new')
catchment = geemap.geopandas_to_ee(catchment_new)


# %% [markdown]
# ***

# %%
def renameBandName(b):
    split = ee.String(b).split('_')   
    return ee.String(split.splice(split.length().subtract(2),1).join("_"))


def buildFeature(i):
    t1 = startDate.advance(i,'day')
    t2 = t1.advance(1,'day')
    #feature = ee.Feature(point)
    dailyColl = collection.filterDate(t1, t2)
    dailyImg = dailyColl.toBands()
    # renaming and handling names
    bands = dailyImg.bandNames()
    renamed = bands.map(renameBandName)
    # Daily extraction and adding time information
    dict = dailyImg.rename(renamed).reduceRegion(
      reducer=ee.Reducer.mean(),
      geometry=catchment,
    ).combine(
      ee.Dictionary({'system:time_start':t1.millis(),'isodate':t1.format('YYYY-MM-dd')})
    )
    return ee.Feature(None,dict)


def getImageCollection(var):
    collection = ee.ImageCollection('NASA/GDDP-CMIP6')\
        .select(var)\
        .filterDate(startDate, endDate)\
        .filterBounds(catchment)
    return collection


def getTask(fileName):
    task = ee.batch.Export.table.toDrive(**{
      'collection': ee.FeatureCollection(ee.List.sequence(0,n).map(buildFeature)),
      'description':fileName,
      'fileFormat': 'CSV'
    })
    return task


# %% [markdown]
# # Set periods for climate scenarios

# %% [markdown]
# To provide the best basis for bias adjustment a large overlap of reanalysis and scenario data is recommended. Per default the routine downloads scenario data starting with the earliest date available from ERA5-Land in 1979 and until 2100.

# %%
start = '1979-01-01'
end = '2100-12-31'                      # exclusive!

# %% [markdown]
# # Launch download tasks for scenario data

# %% [markdown]
# CMIP6 scenario runs start in 2015.

# %%
startDate = ee.Date('2015-01-01')
endDate = ee.Date(end)
n = endDate.difference(startDate,'day').subtract(1)

# %%
collection = getImageCollection('tas')
task_tas_ssp = getTask('CMIP6_tas_ssp')
task_tas_ssp.start()

collection = getImageCollection('pr')
task_pr_ssp = getTask('CMIP6_pr_ssp')
task_pr_ssp.start()

print('Tasks for scenarios started...')

# %% [markdown]
# # Launch download tasks for historical data

# %% [markdown]
# **Caution** - Depending on the selected period, the number of models, Googles server utilization, and other mysterious factors this might take some time. The downloaded files will not exceed 100MB so the bandwidth should not be the problem. As a rough estimate you can plan with 45min for all 122 years and 34 models. In the meantime you can continue in the [next notebook](http://localhost:8888/lab/tree/Seafile/EBA-CA/Repositories/matilda_edu/MATILDA.ipynb) with calibrating MATILDA.

# %% [markdown]
# The CMIP6 historical runs are available for the period of 1950 through 2014.

# %%
startDate = ee.Date(start)
endDate = ee.Date('2014-12-31')
n = endDate.difference(startDate,'day').subtract(1)

# %%
collection = getImageCollection('tas')
task_tas_hist = getTask('CMIP6_tas_hist')
task_tas_hist.start()

collection = getImageCollection('pr')
task_pr_hist = getTask('CMIP6_pr_hist')
task_pr_hist.start()

print('Tasks for historical data started...')

# %% [markdown]
# Start status animation

# %%
import time

while task_tas_ssp.active() or task_pr_ssp.active() or task_tas_hist.active() or task_pr_hist.active():
    print(".", end = '')
    time.sleep(2)
    
print('done.')

# %% [markdown]
# # Wie bekommen wir die Daten vom Drive in den Binder?

# %% [markdown]
# - create ID
# - direct download to new folder with ID name
# - donload the whole folder using the gdown package

# %%
#import gdown

#url = 'https://drive.google.com/drive/folders/1PHEZMh-hJrOS305qHYBCICWap91ITBIE?usp=share_link'
#gdown.download_folder(url, quiet=True, use_cookies=False)

# %% [markdown]
# # Test der neuen Download-Routine mit parallelen Download-Requests

# %% tags=["hide-input"]
import multiprocessing
import geopandas as gpd
import concurrent.futures
import os
import requests
from retry import retry
from tqdm import tqdm


class CMIPDownloader:
    """Class to download spatially averaged CMIP6 data for a given period, variable, and spatial subset."""

    def __init__(self, var, starty, endy, shape, processes=10, dir='./'):
        self.var = var
        self.starty = starty
        self.endy = endy
        self.shape = shape
        self.processes = processes
        self.directory = dir

        # create the download directory if it doesn't exist
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def download(self):
        """Runs a subset routine for CMIP6 data on GEE servers to create ee.FeatureCollections for all years in
        the requested period. Downloads individual years in parallel processes to increase the download time."""

        def getRequests(starty, endy):
            """Generates a list of years to be downloaded. [Client side]"""

            return [i for i in range(starty, endy+1)]

        @retry(tries=10, delay=1, backoff=2)
        def getResult(index, year):
            """Handle the HTTP requests to download one year of CMIP6 data. [Server side]"""

            start = str(year) + '-01-01'
            end = str(year + 1) + '-01-01'
            startDate = ee.Date(start)
            endDate = ee.Date(end)
            n = endDate.difference(startDate, 'day').subtract(1)

            def getImageCollection(var):
                """Create and image collection of CMIP6 data for the requested variable, period, and region.
                [Server side]"""

                collection = ee.ImageCollection('NASA/GDDP-CMIP6') \
                    .select(var) \
                    .filterDate(startDate, endDate) \
                    .filterBounds(self.shape)
                return collection

            def renameBandName(b):
                """Edit variable names for better readability. [Server side]"""

                split = ee.String(b).split('_')
                return ee.String(split.splice(split.length().subtract(2), 1).join("_"))

            def buildFeature(i):
                """Create an area weighted average of the defined region for every day in the given year.
                [Server side]"""

                t1 = startDate.advance(i, 'day')
                t2 = t1.advance(1, 'day')
                # feature = ee.Feature(point)
                dailyColl = collection.filterDate(t1, t2)
                dailyImg = dailyColl.toBands()
                # renaming and handling names
                bands = dailyImg.bandNames()
                renamed = bands.map(renameBandName)
                # Daily extraction and adding time information
                dict = dailyImg.rename(renamed).reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=self.shape,
                ).combine(
                    ee.Dictionary({'system:time_start': t1.millis(), 'isodate': t1.format('YYYY-MM-dd')})
                )
                return ee.Feature(None, dict)

            # Create features for all days in the respective year. [Server side]
            collection = getImageCollection(self.var)
            year_feature = ee.FeatureCollection(ee.List.sequence(0, n).map(buildFeature))

            # Create a download URL for a CSV containing the feature collection. [Server side]
            url = year_feature.getDownloadURL()

            # Handle downloading the actual csv for one year. [Client side]
            r = requests.get(url, stream=True)
            if r.status_code != 200:
                r.raise_for_status()
            filename = os.path.join(self.directory, 'cmip6_' + self.var + '_' + str(year) + '.csv')
            with open(filename, 'w') as f:
                f.write(r.text)

            return index

        # Create a list of years to be downloaded. [Client side]
        items = getRequests(self.starty, self.endy)

        # Launch download requests in parallel processes and display a status bar. [Client side]
        with tqdm(total=len(items), desc="Downloading CMIP6 data for variable '" + self.var + "'") as pbar:
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.processes) as executor:
                for i, year in enumerate(items):
                    results.append(executor.submit(getResult, i, year))
                for future in concurrent.futures.as_completed(results):
                    index = future.result()
                    pbar.update(1)

        print("All downloads complete.")


# %%
cmip_dir = dir_output + 'cmip6/'
downloader_t = CMIPDownloader('tas', 1979, 2100, catchment, processes=25, dir=cmip_dir)
downloader_t.download()
downloader_p = CMIPDownloader('pr', 1979, 2100, catchment, processes=25, dir=cmip_dir)
downloader_p.download()

# %% [markdown]
# # Process the downloaded CSV files --> only works for full period (1979-2100) so far

# %%
import pandas as pd

class CMIPProcessor:
    """Class to read and pre-process CSV files downloaded by the CMIPDownloader class."""
    def __init__(self, var, dir='.'):
        self.dir = dir
        self.var = var
        self.df_hist = self.append_df(self.var, self.dir, hist=True)
        self.df_ssp = self.append_df(self.var, self.dir, hist=False)
        self.ssp2_common, self.ssp5_common, self.hist_common,\
            self.common_models, self.dropped_models = self.process_dataframes()
        self.ssp2, self.ssp5 = self.get_results()

    def read_cmip(self, filename):
        """Reads CMIP6 CSV files and drops redundant columns."""

        df = pd.read_csv(filename, index_col='isodate', parse_dates=['isodate'])
        df = df.drop(['system:index', '.geo', 'system:time_start'], axis=1)
        return df

    def append_df(self, var, dir='.', hist=True):
        """Reads CMIP6 CSV files of individual years and concatenates them into dataframes for the full downloaded
        period. Historical and scenario datasets are treated separately. Drops a model with data gaps.
        Converts precipitation unit to mm."""

        df_list = []
        if hist:
            starty = 1979
            endy = 2014
        else:
            starty = 2015
            endy = 2100
        for i in range(starty, endy + 1):
            filename = dir + 'cmip6_' + var + '_' + str(i) + '.csv'
            df_list.append(self.read_cmip(filename))
        if hist:
            hist_df = pd.concat(df_list).drop('historical_GFDL-CM4_' + var, axis=1)
            if var == 'pr':
                hist_df = hist_df * 86400       # from kg/(m^2*s) to mm/day
            return hist_df
        else:
            ssp_df = pd.concat(df_list).drop(['ssp585_GFDL-CM4_' + var, 'ssp245_GFDL-CM4_' + var], axis=1)
            if var == 'pr':
                ssp_df = ssp_df * 86400       # from kg/(m^2*s) to mm/day
            return ssp_df

    def process_dataframes(self):
        """Separates the two scenarios and drops models not available for both scenarios and the historical period."""

        ssp2 = self.df_ssp.loc[:, self.df_ssp.columns.str.startswith('ssp245')]
        ssp5 = self.df_ssp.loc[:, self.df_ssp.columns.str.startswith('ssp585')]
        hist = self.df_hist.loc[:, self.df_hist.columns.str.startswith('historical')]

        ssp2.columns = ssp2.columns.str.lstrip('ssp245_').str.rstrip('_' + self.var)
        ssp5.columns = ssp5.columns.str.lstrip('ssp585_').str.rstrip('_' + self.var)
        hist.columns = hist.columns.str.lstrip('historical_').str.rstrip('_' + self.var)

        # Get all the models the three datasets have in common
        common_models = set(ssp2.columns).intersection(ssp5.columns).intersection(hist.columns)

        # Get the model names that contain NaN values
        nan_models_list = [df.columns[df.isna().any()].tolist() for df in [ssp2, ssp5, hist]]
        # flatten the list
        nan_models = [col for sublist in nan_models_list for col in sublist]
        # remove duplicates
        nan_models = list(set(nan_models))

        # Remove models with NaN values from the list of common models
        common_models = [x for x in common_models if x not in nan_models]

        ssp2_common = ssp2.loc[:, common_models]
        ssp5_common = ssp5.loc[:, common_models]
        hist_common = hist.loc[:, common_models]

        dropped_models = list(set([mod for mod in ssp2.columns if mod not in common_models] +
                                  [mod for mod in ssp5.columns if mod not in common_models] +
                                  [mod for mod in hist.columns if mod not in common_models]))

        return ssp2_common, ssp5_common, hist_common, common_models, dropped_models

    def get_results(self):
        """Concatenates historical and scenario data to combined dataframes of the full downloaded period."""

        ssp2_full = pd.concat([self.hist_common, self.ssp2_common])
        ssp2_full.index.names = ['TIMESTAMP']
        ssp5_full = pd.concat([self.hist_common, self.ssp5_common])
        ssp5_full.index.names = ['TIMESTAMP']

        return ssp2_full, ssp5_full


# %%
## Usage example
processor = CMIPProcessor(dir=cmip_dir, var='pr')
ssp2_pr, ssp5_pr = processor.get_results()
processor = CMIPProcessor(dir=cmip_dir, var='tas')
ssp2_tas, ssp5_tas = processor.get_results()

print(ssp2_tas)

# %%
# Test edit
