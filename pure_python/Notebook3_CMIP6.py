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
# # Climate scenario data

# %% [markdown]
# In this notebook we will...
# 1. ... aggregate and download climate scenario data from the Coupled Model Intercomparison Project Phase 6 ([CMIP6](https://wcrp-cmip.org/cmip-phase-6-cmip6/)) for our catchment,
# 2. ... preprocess the data,
# 3. ... compare the CMIP6 models with our reanalysis data and adjust them for biases,
# 4. ... and visualize the data before and after bias adjustment.
#
# The [NEX-GDDP-CMIP6 dataset](https://www.nature.com/articles/s41597-022-01393-4) we are going to use has been downscaled to 27830 m resolution by the [NASA Climate Analytics Group](https://www.nature.com/articles/s41597-022-01393-4) and is available in two [Shared Socio-Economic Pathways](https://unfccc.int/sites/default/files/part1_iiasa_rogelj_ssp_poster.pdf) (SSP2 and SSP5). It is available via [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/NASA_GDDP-CMIP6#bands) which makes it subsettable on the server side and the download files relatively lightweight.

# %% [markdown]
# We start by reading the config and initializing the Google Earth Engine access again.

# %%
import warnings
warnings.filterwarnings("ignore", category=UserWarning)     # Suppress Deprecation Warnings
import configparser
import ast
import geopandas as gpd
import ee
import geemap
import numpy as np


# read local config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

# get paths from config.ini
dir_output = config['FILE_SETTINGS']['DIR_OUTPUT']
dir_figures = config['FILE_SETTINGS']['DIR_FIGURES']
output_gpkg = dir_output + config['FILE_SETTINGS']['GPKG_NAME']
zip_output = config['CONFIG']['ZIP_OUTPUT']

# get style for matplotlib plots
# plt_style = ast.literal_eval(config['CONFIG']['PLOT_STYLE'])

# set the file format for storage
compact_files = config.getboolean('CONFIG','COMPACT_FILES')

# read cloud-project
cloud_project = config['CONFIG']['CLOUD_PROJECT']

# initialize GEE
try:
    ee.Initialize(project=cloud_project)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=cloud_project)

# %% [markdown]
# The next cell reads the output directory location and the catchment outline as target polygon.

# %%
# load catchment outline as target polygon
catchment_new = gpd.read_file(output_gpkg, layer='catchment_new')
catchment = geemap.geopandas_to_ee(catchment_new)

# name target subdirectory to be created
cmip_dir = dir_output + 'cmip6/'

# %% [markdown]
# ## Select, aggregate, and download downscaled CMIP6 data
# %% [markdown]
# We have designed a class called `CMIPDownloader` that does all this in one go. The `buildFeature()` function requests daily catchment wide averages of all available CMIP6 models for individual years. All requested years are stored in an `ee.ImageCollection` by `getResult()`. To provide the best basis for bias adjustment, a large overlap of reanalysis and scenario data is recommended. By default, the `CMIPDownloader` class requests everything between the earliest available date from ERA5 (1979) and the latest available date from CMIP6 (2100). The `download()` function then starts a given number of parallel requests, each downloading a single year and saving it as a CSV file.
#
# We can simply specify a target location and start the download for both variables individually. We choose a moderate number of requests to avoid kernel hickups. The download time depends on the number of parallel processes, the traffic on the GEE servers and other mysterious factors. If you run this notebook in a binder, it usually doesn't take more than 5 minutes for both downloads to finish.

# %%
from tools.geetools import CMIPDownloader

downloader_t = CMIPDownloader(var='tas', starty=1979, endy=2100, shape=catchment, processes=30, dir=cmip_dir)
downloader_t.download()
downloader_p = CMIPDownloader(var='pr', starty=1979, endy=2100, shape=catchment, processes=30, dir=cmip_dir)
downloader_p.download()

# %% [markdown]
# We have now downloaded individual files for each year and variable and stored them in `cmip_dir`. To use them as model forcing data, they need to be processed.

# %% [markdown]
# ## Process the downloaded CSV files

# %% [markdown]
# The corresponding `CMIPProcessor` class will read all downloaded CSV files and concatenate them into a single file per scenario. It also checks for consistency and drops models that are not available for individual years or scenarios. It processes variables individually and returns a single data frame for each of the two scenarios from 1979 to 2100.

# %%
from tools.geetools import CMIPProcessor

cmip_dir = dir_output + 'cmip6/'

processor_t = CMIPProcessor(file_dir=cmip_dir, var='tas')
ssp2_tas_raw, ssp5_tas_raw = processor_t.get_results()

processor_p = CMIPProcessor(file_dir=cmip_dir, var='pr')
ssp2_pr_raw, ssp5_pr_raw = processor_p.get_results()

# %% [markdown]
# Let's have a look. We can see that our scenario dataset now contains 33 CMIP6 models in alphabetical order.

# %% tags=["output_scroll"]
print(ssp2_tas_raw.info())

# %% [markdown]
# If we want to check which models failed the consistency check of the `CMIPProcessor` we can use its `dropped_models` attribute.

# %%
print('Models that failed the consistency checks:\n')
print(processor_t.dropped_models)

# %% [markdown]
# ## Bias adjustment using reananlysis data

# %% [markdown]
# Due to the coarse resolution of global climate models (GCMs) and the extensive correction of reanalysis data there is substantial bias between the two datasets. To force a glacio-hydrological model calibrated on reanalysis data with climate scenarios this bias needs to be adressed. We will use a method developed by [Switanek et.al. (2017)](https://doi.org/10.5194/hess-21-2649-2017) called Scaled Distribution Mapping (SDM) to correct for bias while preserving trends and the likelihood of meteorological events in the raw GCM data. The method has been implemented in the [`bias_correction`](https://github.com/pankajkarman/bias_correction) Python library by [Pankaj Kumar](https://pankajkarman.github.io/). As suggested by the authors we will apply the bias adjustment to discrete blocks of data individually.
# We will first create a function to read our reanalysis CSV. The `adjust_bias()` function will then loop over all models and adjust them to the reanalysis data in the overlap period (1979 to 2022).

# %%
from tools.helpers import read_era5l
from bias_correction import BiasCorrection
import pandas as pd

def adjust_bias(predictand, predictor, method='normal_mapping'):
    """Applies bias correction to discrete periods individually."""
    # Read predictor data
    predictor = read_era5l(predictor)

    # Determine variable type based on the mean value
    var = 'temp' if predictand.mean().mean() > 100 else 'prec'

    # Adjust bias in discrete blocks as suggested by Switanek et al. (2017)
    correction_periods = [
        {'correction_range': ('1979-01-01', '2010-12-31'), 'extraction_range': ('1979-01-01', '1990-12-31')},
    ]
    for decade_start in range(1991, 2090, 10):
        correction_start = f"{decade_start - 10}-01-01"
        correction_end = f"{decade_start + 19}-12-31"
        extraction_start = f"{decade_start}-01-01"
        extraction_end = f"{decade_start + 9}-12-31"

        correction_periods.append({
            'correction_range': (correction_start, correction_end),
            'extraction_range': (extraction_start, extraction_end)
        })

    correction_periods.append({
        'correction_range': ('2081-01-01', '2100-12-31'),
        'extraction_range': ('2091-01-01', '2100-12-31')
    })

    # Store corrected periods
    corrected_data_list = []
    training_period = slice('1979-01-01', '2022-12-31')

    for period in correction_periods:
        correction_start, correction_end = period['correction_range']
        extraction_start, extraction_end = period['extraction_range']

        correction_slice = slice(correction_start, correction_end)
        extraction_slice = slice(extraction_start, extraction_end)

        data_corr = pd.DataFrame()
        for col in predictand.columns:
            x_train = predictand[col][training_period].squeeze()
            y_train = predictor[training_period][var].squeeze()
            x_predict = predictand[col][correction_slice].squeeze()
            bc_corr = BiasCorrection(y_train, x_train, x_predict)
            corrected_col = pd.DataFrame(bc_corr.correct(method=method))
            data_corr[col] = corrected_col.loc[extraction_slice]

        corrected_data_list.append(data_corr)

    corrected_data = pd.concat(corrected_data_list, axis=0)
    return corrected_data


# %% [markdown]
# The function is applied separately to each variable and scenario. The `bias_adjustment` library provides a normal and a gamma distribution as a basis for the SDM. As the distribution of the ERA5 Land precipitation data is actually closer to a normal distribution with a cut-off of 0 mm, we use the `normal_mapping` method for both variables.

# %%
era5_file = dir_output + 'ERA5L.csv'

ssp2_tas = adjust_bias(predictand=ssp2_tas_raw, predictor=era5_file)
ssp5_tas = adjust_bias(predictand=ssp5_tas_raw, predictor=era5_file)
ssp2_pr = adjust_bias(predictand=ssp2_pr_raw, predictor=era5_file)
ssp5_pr = adjust_bias(predictand=ssp5_pr_raw, predictor=era5_file)

# %% [markdown]
# The result is a comprehensive dataset of 33 models over 122 years in two versions (pre- and post-adjustment) for every variable. To see what's in the data and what happened during bias adjustment we need an overview.

# %% [markdown]
# ## 	Visualization

# %% [markdown]
# First, we store our raw and adjusted data in dictionaries.

# %%
ssp_tas_dict = {'SSP2_raw': ssp2_tas_raw, 'SSP2_adjusted': ssp2_tas, 'SSP5_raw': ssp5_tas_raw, 'SSP5_adjusted': ssp5_tas}
ssp_pr_dict = {'SSP2_raw': ssp2_pr_raw, 'SSP2_adjusted': ssp2_pr, 'SSP5_raw': ssp5_pr_raw, 'SSP5_adjusted': ssp5_pr}

# %% [markdown]
# The first plot will contain simple timeseries. The first function `cmip_plot()` resamples the data so a given frequency and creates a single plot. `cmip_plot_combined()` arranges multiple plots for both scenarios before and after bias adjustment.

# %% [markdown]
# ### Time series

# %% [markdown]
# By default, the data is smoothed with a 10-year moving average (`smooting_window=10`). Precipitation data is aggregated to annual totals (`agg_level='annual'`). You can customise this by specifying the appropriate arguments.

# %%
from tools.plots import cmip_plot_combined

era5 = read_era5l(era5_file)

cmip_plot_combined(data=ssp_tas_dict, target=era5, title='10y Rolling Mean of Daily Air Temperature', target_label='ERA5-Land', smooth_window=10, show=True, fig_path=f"{dir_figures}NB3_CMIP6_Temp.png")
cmip_plot_combined(data=ssp_pr_dict, target=era5, title='10y Rolling Mean of Annual Precipitation', precip=True, target_label='ERA5-Land', agg_level='annual', smooth_window=10, show=True, fig_path=f"{dir_figures}NB3_CMIP6_Prec.png")

# %% [markdown]
# Apparently, some models have striking curves indicating unrealistic outliers or sudden jumps in the data. To clearly identify faulty time series, one option is to choose a qualitative approach by identifying the models using an interactive `plotly` plot. Here we can zoom and select/deselect curves as we like, to identify model names.

# %%
import plotly.express as px

fig = px.line(ssp5_tas_raw.resample('10YE', closed='left', label='left').mean())
fig.show()


# %% [markdown]
# ### Violin plots

# %% [markdown]
# To look at it from a different perpective we can also have a look at the individual distributions of all models. A nice way to cover several aspects at once is to use `seaborne` [violinplots](https://seaborn.pydata.org/generated/seaborn.violinplot.html).
#
# First we have to rearrange our input dictionaries a little bit. 

# %%
def dict_filter(dictionary, filter_string):
    """Returns a dict with all elements of the input dict that contain a filter string in their keys."""
    return {key.split('_')[0]: value for key, value in dictionary.items() if filter_string in key}


tas_raw = dict_filter(ssp_tas_dict, 'raw')
tas_adjusted = dict_filter(ssp_tas_dict, 'adjusted')
pr_raw = dict_filter(ssp_pr_dict, 'raw')
pr_adjusted = dict_filter(ssp_pr_dict, 'adjusted')

# %% [markdown]
# For comparison the `vplots()` function will arrange the plots in a similar grid as in the figures above.

# %%
from tools.plots import vplots

vplots(tas_raw, tas_adjusted, era5, target_label='ERA5-Land', show=True, fig_path=f"{dir_figures}NB3_vplot_Temp.png")
vplots(pr_raw, pr_adjusted, era5, target_label='ERA5-Land', precip=True, show=True, fig_path=f"{dir_figures}NB3_vplot_Prec.png")

# %% [markdown]
# ### Data filters

# %% [markdown]
# Since we have a large number of models and some problems may be difficult to identify in a plot, we can use some standard filters combined in the `DataFilter` class. By default it filters models that contain ...
#
#   ... outliers deviating more than 3 standard deviations from the mean (`zscore_threshold`) and/or ...
#
#   ... increases/decreases of more than 5 units in a single timestep (`jump_threshold`).
#
# The functions can be applied separately (`check_outliers` or `check_jumps`) or together (`filter_all`). All three return a `list` of model names.
#
# Here, we also use the `resampling_rate` parameter to resample the data to annual means (`'YE'`) before running the checks.

# %%
from tools.helpers import DataFilter

filter = DataFilter(ssp5_tas_raw, zscore_threshold=3, jump_threshold=5, resampling_rate='YE')

print('Models with temperature outliers: ' + str(filter.outliers))
print('Models with temperature jumps: ' + str(filter.jumps))
print('Models with either one or both: ' + str(filter.filtered_models))

# %% [markdown]
# The identified columns can then be removed from the CMIP6 ensemble dataset using another helper function. We run the `drop_model()` function on the dictionaries of all variables and run `cmip_plot_combined()` again to check the result.

# %%
from tools.helpers import drop_model

ssp_tas_dict = drop_model(filter.filtered_models, ssp_tas_dict)
ssp_pr_dict = drop_model(filter.filtered_models, ssp_pr_dict)


cmip_plot_combined(data=ssp_tas_dict, target=era5, title='10y Mean of Air Temperature', target_label='ERA5-Land', show=True, fig_path=f"{dir_figures}NB3_CMIP6_Temp_filtered.png")
cmip_plot_combined(data=ssp_pr_dict, target=era5, title='10y Mean of Monthly Precipitation', precip=True, target_label='ERA5-Land', show=True, agg_level='annual', smooth_window=10, fig_path=f"{dir_figures}NB3_CMIP6_Prec_filtered.png")

# %% [markdown]
# ### Ensemble mean plots

# %% [markdown]
# As we now don't need to focus on individual models anymore, we can reduce the number of lines by only plotting the ensemble means with a 90% confidence interval. With less lines in the plot, we can also reduce the resample frequency and show annual means.

# %%
from tools.plots import cmip_plot_ensemble

cmip_plot_ensemble(ssp_tas_dict, era5['temp'], intv_mean='YE', fig_path=f'{dir_figures}NB3_CMIP6_Ensemble_Temp.png')
cmip_plot_ensemble(ssp_pr_dict, era5['prec'], precip=True, intv_sum='ME', intv_mean='YE', fig_path=f'{dir_figures}NB3_CMIP6_Ensemble_Prec.png')

# %% [markdown]
# We can see that the SDM adjusts the range and mean of the target data while preserving the distribution and trend of the original data. However, the inter-model variance is slightly reduced for temperature and significantly increased for precipitation.

# %% [markdown]
# Last but not least, we will have a closer look at the performance of the bias adjustment. To do that, we will create probability plots for all models comparing original, target, and adjusted data with each other and a standard normal distribution. The `prob_plot` function creates such a plot for an individual model and scenario. The `pp_matrix` function loops the `prob_plot` function over all models in a `DataFrame` and arranges them in a matrix.

# %% [markdown]
# First we'll have a look at the temperature.

# %%
from tools.plots import pp_matrix

pp_matrix(ssp2_tas_raw, era5['temp'], ssp2_tas, scenario='SSP2', show=True, fig_path=f'{dir_figures}NB3_CMIP6_SSP2_probability_Temp.png')
pp_matrix(ssp5_tas_raw, era5['temp'], ssp5_tas, scenario='SSP5', show=True, fig_path=f'{dir_figures}NB3_CMIP6_SSP5_probability_Temp.png')

# %% [markdown]
# We can see that the SDM worked very well for the temperature data, with high agreement between the target and adjusted data.
#
# Let's look at the probability curves for precipitation. Since the precipitation data is bounded at 0, but most days have very small values >0 mm, we resample the data to monthly sums to get an idea of the overall performance.

# %%
pp_matrix(ssp2_pr_raw, era5['prec'], ssp2_pr, precip=True, scenario='SSP2', show=True, fig_path=f'{dir_figures}NB3_CMIP6_SSP2_probability_Prec.png')
pp_matrix(ssp5_pr_raw, era5['prec'], ssp5_pr, precip=True, scenario='SSP5', show=True, fig_path=f'{dir_figures}NB3_CMIP6_SSP5_probability_Prec.png')


# %% [markdown]
# Considering the complexity and heterogeneity of precipitation data, the performance of SDM is convincing. While the fitted data of most models deviate from the target data for low and very high values, the general distribution of monthly precipitation is well met. 

# %% [markdown]
# ## Write CMIP6 data to file

# %% [markdown]
# After a thorough review of the climate scenario data, we can write the final selection to files to be used in the next notebook. We want to use reanalysis data for the MATILDA model wherever possible and only use CMIP6 data for future projections. Therefore, we need to replace all of the data from our calibration period with ERA5-Land data.

# %%
def replace_values(target_df, source_df, source_column):
    """
    Replaces values in the overlapping period in the target dataframe with values
    from the source dataframe using the specified source column.

    Args:
        target_df (pd.DataFrame): Target dataframe where values will be replaced.
        source_df (pd.DataFrame): Source dataframe from which values will be taken.
        source_column (str): Column name in the source dataframe to use for replacement.

    Returns:
        pd.DataFrame: The target dataframe with updated values.
    """

    # Identify overlapping period based on index (datetime)
    overlapping_period = target_df.index.intersection(source_df.index)


    if len(overlapping_period) == 0:
        raise ValueError("No overlapping period between the source and target dataframes.")

    # Ensure the source dataframe has the required column
    if source_column not in source_df.columns:
        raise ValueError(f"The source dataframe does not have a column named '{source_column}'")
    
    # Get the replacement values from the source columnAdd commentMore actions
    replacement_values = source_df.loc[overlapping_period, source_column]

    assert len(overlapping_period) == len(
        replacement_values), "Mismatch in lengths of overlapping period and replacement values."

    # Apply these values to all columns in the target DataFrame in the overlapping period
    target_df.loc[overlapping_period] = replacement_values.values[:, None]

    return target_df


era5l = read_era5l(era5_file)
ssp2_tas = ssp_tas_dict['SSP2_adjusted'].copy()
ssp5_tas = ssp_tas_dict['SSP5_adjusted'].copy()
ssp2_pr = ssp_pr_dict['SSP2_adjusted'].copy()
ssp5_pr = ssp_pr_dict['SSP5_adjusted'].copy()

ssp2_tas = replace_values(ssp2_tas, era5l, 'temp')
ssp5_tas = replace_values(ssp5_tas, era5l, 'temp')
ssp2_pr = replace_values(ssp2_pr, era5l, 'prec')
ssp5_pr = replace_values(ssp5_pr, era5l, 'prec')

# %% [markdown]
# Since the whole ensemble results in relatively large files, we store the dictionaries in binary format. While these are not human-readable, they are compact and fast to read and write.

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Note:</b> In the config file you can choose between two storage options: <code>pickle</code> files are fast to read and write, but take up more disk space (<code>COMPACT_FILES = False</code>). You can use them on your local machine. <code>parquet</code> files need less disk space but take longer to read and write (<code>COMPACT_FILES = True</code>). They should be your choice in the Binder.</div>

# %%
from tools.helpers import dict_to_pickle, dict_to_parquet

tas = {'SSP2': ssp2_tas, 'SSP5': ssp5_tas}
pr = {'SSP2': ssp2_pr, 'SSP5': ssp5_pr}

if compact_files:
    # For storage efficiency:
    dict_to_parquet(tas, cmip_dir + 'adjusted/tas_parquet')
    dict_to_parquet(pr, cmip_dir + 'adjusted/pr_parquet')
else:
    # For speed:
    dict_to_pickle(tas, cmip_dir + 'adjusted/tas.pickle')
    dict_to_pickle(pr, cmip_dir + 'adjusted/pr.pickle')

# %%
import shutil

if zip_output:
    # refresh `output_download.zip` with data retrieved within this notebook
    shutil.make_archive('output_download', 'zip', 'output')
    print('Output folder can be download now (file output_download.zip)')

# %%
# %reset -f
