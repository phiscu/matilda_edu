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
# # Calibrating the MATILDA framework

# %% [markdown]
# In this notebook we will
#
# 1. ... set up a glacio-hydrological model with all the data we have collected,
#
# 2. ... run the model for the calibration period with default parameters and check the results,
#
# 3. ... use a statistical parameter optimization routine to calibrate the model,
#
# 4. ... and store the calibrated parameter set for the scenario runs in the next notebook.
#
#
# The framework for Modeling water resources in glacierized catchments [MATILDA] (https://github.com/cryotools/matilda) has been developed for use in this workflow and is published as a Python package. It is based on the widely used [HBV hydrological model](https://www.cabdirect.org/cabdirect/abstract/19961904773), extended by a simple temperature-index melt model based  on the code of [Seguinot (2019)](https://zenodo.org/record/3467639). Glacier evolution over time is modeled using a modified version of the &Delta;*h* approach following [Seibert et. al. (2018)](https://doi.org/10.5194/hess-22-2211-2018).

# %% [markdown]
# As before we start by loading configurations such as the calibration period and some helper functions to work with  `yaml` files.


# %%
from tools.helpers import update_yaml, read_yaml, write_yaml
import configparser
import ast

# read local config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

# get output dir and date range from config.ini
dir_input = config['FILE_SETTINGS']['DIR_INPUT']
dir_output = config['FILE_SETTINGS']['DIR_OUTPUT']
date_range = ast.literal_eval(config['CONFIG']['DATE_RANGE'])

print('MATILDA will be calibrated on the period ' + date_range[0] + ' to ' + date_range[1])

# %% [markdown]
# Since HBV is a so-called 'bucket' model and all the buckets are empty in the first place we need to fill them in a setup period of minimum one year. If not specified, the first two years of the `DATE_RANGE` in the `config` are used for set up.

# %%
import pandas as pd

length_of_setup_period = 2

sim_start = pd.to_datetime(date_range[0]) + pd.DateOffset(years = length_of_setup_period)
set_up_end = sim_start - pd.DateOffset(days = 1)

dates = {'set_up_start': date_range[0],
         'set_up_end': str(set_up_end).split(' ')[0],        # remove hh:mm:ss
         'sim_start': str(sim_start).split(' ')[0],          # remove hh:mm:ss
         'sim_end': date_range[1]}

for key in dates.keys(): print(key + ': ' + dates[key])

# %% [markdown]
# Many MATILDA parameters have been calculated in previous notebooks and stored in `settings.yaml`. We can easily add the modeling periods using a helper function. The calculated glacier profile from Notebook 1 can be imported as a `pandas DataFrame` and added to the `settings` dictionary as well.
#
# Finally, we will also add some optional settings that control the aggregation frequency of the outputs, the choice of graphical outputs, and more.

# %% tags=["output_scroll"]
update_yaml(dir_output + 'settings.yml', dates)

remaining_settings = {"freq": "M",               # aggregation frequency of model outputs (D, M, Y)
                      "warn": False,             # show warnings of subpackages?
                      "plot_type": "all",        # interactive and/or non-interactive plots ('print', 'interactive', 'all')
                      "elev_rescaling": True}    # treat mean glacier elevation as constant or change with glacier evolution

update_yaml(dir_output + 'settings.yml', remaining_settings)

settings = read_yaml(dir_output + 'settings.yml')
glacier_profile = pd.read_csv(dir_output + 'glacier_profile.csv')
settings['glacier_profile'] = glacier_profile

print('MATILDA settings:\n\n')
for key in settings.keys(): print(key + ': ' + str(settings[key]))

# %% [markdown]
# ## Run MATILDA with default parameters

# %% [markdown]
# We will force MATILDA with the pre-processed ERA5-Land data from Notebook 2. Although MATILDA can run without calibration on observations, the results would have extreme uncertainties. Therefore, we recommend to use at least discharge observations for your selected point to evaluate the simulations against. Here, we load discharge observations for your example catchment. As you can see, we have meteorological forcing data since 1979 and discharge data since 1982. However, most of the glacier datasets used start in 2000/2001 and glaciers are a significant part of the water balance. In its current version, MATILDA is not able to extrapolate glacier cover, so we will only calibrate the model using data from 2000 onwards.

# %% tags=["output_scroll"]
era5 = pd.read_csv(dir_output + 'ERA5L.csv', usecols=['dt', 'temp', 'prec'])
era5.columns = ['TIMESTAMP', 'T2', 'RRR']

# remove HH:MM:SS from 'TIMESTAMP' column
era5['TIMESTAMP'] = pd.to_datetime(era5['TIMESTAMP'])
era5['TIMESTAMP'] = era5['TIMESTAMP'].dt.date

print('ERA5 Data:')
display(era5)

obs = pd.read_csv(dir_input + 'obs_runoff_example.csv')

print('Observations:')
display(obs)

# %% [markdown]
# With all settings and input data in place, we can run MATILDA with **default parameters**.

# %% tags=["output_scroll"]
from matilda.core import matilda_simulation

output_matilda = matilda_simulation(era5, obs, **settings)

# %% [markdown]
# The results are obviously far from reality and largely overestimate runoff. The **Kling-Gupta Efficiency coefficient ([KGE](https://doi.org/10.1029/2011WR010962))** rates the result as 0.36 with 1.0 being a perfect match with the observations. We can also see that the input precipitation is much higher than the total runoff. Clearly, **the model needs calibration**.

# %% [markdown]
# ## Calibrate MATILDA

# %% [markdown]
# In order to adjust the model parameters to the catchment characteristics, we will perform an **automated calibration**. The [MATILDA Python package](https://github.com/cryotools/matilda) contains a calibration module called *[matilda.mspot](https://github.com/cryotools/matilda/blob/master/matilda/mspot_glacier.py)* that makes extensive use of the Statistical Parameter Optimization Tool for Python ([SPOTPY](https://doi.org/10.1371/journal.pone.0145180)). As we have seen, there can be large uncertainties in the input data (especially precipitation). Simply tuning the model parameters to match the observed hydrograph may over- or underestimate other runoff contributors, such as glacier melt, to compensate for deficiencies in the precipitation data. Therefore, it is good practice to include additional data sets in a **multi-objective calibration**. In this workflow we will use:
#
# - remote sensing estimates of **glacier surface mass balance** (**SMB**) and ...
#
# - **snow water equivalent** (**SWE**) estimates from a dedicated snow reanalysis product.
#
# Unfortunately, both default datasets are **limited to High Mountain Asia (HMA)**. For study sites in other areas, please consult other sources and manually add the target values for calibration in the appropriate code cells. We are happy to include additional datasets as long as they are available online and can be integrated into the workflow.
#
# <div class="alert alert-block alert-info">
# <b>Note:</b> Statistical parameter optimization (SPOT) algorithms require a high number of model runs, especially for large parameter sets. Both <i>mybinder.org</i> and <i>Google Colab</i> offer a maximum of two cores per user. One MATILDA calibration run for 20 years takes roughly 3s on one core. Therefore, large optimization runs in an online environment will be slow and may require you to leave the respective browser tab in the foreground for hours. To speed things up, you can either...</div>
#
# ... run this notebook **locally on a computer with more cores** (ideally a high performance cluster) or ...
#
# ... **reduce the number of calibration parameters** based the global sensitivity. We will return to this topic later in this notebook.
#
# For now, we will demonstrate how to use the SPOT features and then continue with a parameter set from a large HPC optimization run. If you need help implementing the routine on your HPC, consult the [SPOTPY documentation](https://spotpy.readthedocs.io/en/latest/Advanced_hints/#mpi-parallel-computing) and [contact us](https://github.com/phiscu/matilda_edu/issues/new) if you encounter problems.

# %% [markdown]
# ### Glacier surface mass balance data

# %% [markdown]
# There are various sources of SMB records but only remote sensing estimates provide data for (almost) all glaciers in the target catchment. [Shean et. al. 2020 ](https://doi.org/10.3389/feart.2019.00363) calculated geodetic mass balances for all glaciers in High Mountain Asia from 2000 to 2018. For this example (and all other catchments in HMA), we can use their data set so derive an average annual mass balance in the calibration period.
#
# <div class="alert alert-block alert-info">
# <b>Note:</b> As all remote sensing estimates the used dataset has significant uncertainties. A comparison to other datasets and the impact on the modeling results are discussed in the associated publication. </div>
#
# We pick all individual mass balances that match the glacier IDs in our catchment and calculate the catchment-wide mean. In addition, we use the uncertainty estimate provided in the dataset to derive an uncertainty range.

# %%
import pandas as pd

mass_balances = pd.read_csv(dir_input + '/hma_mb_20190215_0815_rmse.csv', usecols=['RGIId', 'mb_mwea', 'mb_mwea_sigma'])
ids = pd.read_csv(dir_output + '/RGI/Glaciers_in_catchment.csv')

merged = pd.merge(mass_balances, ids, on='RGIId')
mean_mb = round(merged['mb_mwea'].mean() * 1000, 3)   # Mean catchment MB in mm w.e.
mean_sigma = round(merged['mb_mwea_sigma'].mean() * abs(mean_mb), 3)  # Mean uncertainty of catchment MB in mm w.e.

target_mb = [mean_mb - mean_sigma, mean_mb + mean_sigma]

print('Target glacier mass balance for calibration: ' + str(mean_mb) + ' +-' + str(mean_sigma) + 'mm w.e.')

# %% [markdown]
# ### Snow water equivalent

# %% [markdown]
# For snow cover estimates, we will use a specialized **snow reanalysis** dataset from [Liu et. al. 2021](https://doi.org/10.5067/HNAUGJQXSCVU). Details on the data can be found in the dataset documentation and the associated [publication](https://doi.org/10.1029/2022GL100082).
# Unfortunately, access to the dataset requires a (free) registration at **NASA's EarthData** portal, which prevents a seamless integration. Also, the dataset consists of large files and requires some pre-processing that **could not be done in a Jupyter Notebook**. However, we provide you with the `SWEETR` tool, a **fully automated workflow** that you can run on your local computer to download, process, and aggregate the data. Please refer to the dedicated [Github repository](https://github.com/phiscu/hma_snow) for further instructions.

# %% [markdown]
# <img src="images/mean_swe_example.png" alt="Mean_SWE" width="400">

# %% [markdown]
# When you run the `SWEETR` tool with your catchment outline, it returns cropped raster files as the one shown above and, more importantly, a **timeseries of catchment-wide daily mean SWE** from 1999 to 2016. Just replace the `input/swe.csv` file with your result. Now we can load the timeseries as a dataframe.

# %%
swe = pd.read_csv(f'{dir_input}/swe.csv')

# %% [markdown]
# Along with the cropped SWE rasters the `SWEETR` tool creates binary masks for seasonal- and non-seasonal snow. Due to its strong signal in remote sensing data, seasonal snow can be better detected leading to more robust SWE estimates. However, the non-seasonal snow largely agrees with the glacierized area. Therefore, we will calibrate the snow routine by comparing the SWE of the ice-free sub-catchment with the seasonal snow of the reanalysis. Since the latter has a coarse resolution of 500 m, the excluded catchment area is a bit larger than the RGI glacier outlines (17.2% non-seasonal snow vs. 10.8% glacierized sub-catchment). Therefore, we use a scaling factor to account for this mismatch in the reference area.

# %%
glac_ratio = settings['area_glac'] / settings['area_cat']       # read glacieriezed and total area from the settings
swe_area_sim = 1-glac_ratio
swe_area_obs = 0.828            # 1 - non-seasonal snow / seasonal snow
sf = swe_area_obs / swe_area_sim
print('SWE scaling factor: ' + str(round(sf, 3)))

# %% [markdown]
# The MATILDA framework provides an interface for [SPOTPY](https://github.com/thouska/spotpy/). Here we will use the `psample()` function to run MATILDA with the same settings as before but varying parameters. To do this, we will remove redundant `settings` and add some new ones specific to the function. Be sure to choose the number of repetitions carefully.

# %% tags=["output_scroll"]
from tools.helpers import drop_keys

psample_settings = drop_keys(settings, ['warn', 'plots', 'plot_type'])

additional_settings = {'rep': 10,                             # Number of model runs. For advice, check the documentation of the algorithms.
                       'glacier_only': False,                 # True when calibrating an entirely glacierized catchment
                       'obj_dir': 'maximize',                 # should your objective function be maximized (e.g. NSE) or minimized (e.g. RMSE)
                       'target_mb': mean_mb,                  # Average annual glacier mass balance to target at
                       'target_swe': swe,                     # Catchment-wide mean SWE timeseries of seasonal snow to calibrate the snow routine
                       'swe_scaling': 0.928,                  # scaling factor for simulated SWE to account for reference area mismatch
                       'dbformat': None,                      # Write the results to a file ('csv', 'hdf5', 'ram', 'sql')
                       'output': None,                        # Choose where to store the files
                       'algorithm': 'lhs',                    # Choose algorithm (for parallelization: mc, lhs, fast, rope, sceua or demcz)
                       'dbname': 'era5_matilda_example',      # Choose name
                       
                       'parallel': False,                     # Distribute the calculation on multiple cores or not
                      # 'cores': 20                           # Set number of cores when running in parallel
                      }
psample_settings.update(additional_settings)

print('Settings for calibration runs:\n')
for key in psample_settings.keys(): print(key + ': ' + str(psample_settings[key]))

# %% [markdown]
# With these settings we can start the `psample()` to run our model with various parameter combinations. The default parameter boundaries can be found in the MATILDA [parameter documentation](https://github.com/cryotools/matilda/tree/master?tab=readme-ov-file#parameter-list). If you want to narrow down the parameter space, you can do that using the following syntax. Here, we define custom ranges for the temperature lapse rate and the precipitation correction factor and run a short Latin Hypercube Sampling (LHS) as an example.

# %% tags=["output_scroll"]
from matilda.mspot_glacier import psample

lim_dict = {'lr_temp_lo': -0.007, 'lr_temp_up': -0.005, 'PCORR_lo': 0.5, 'PCORR_up': 1.5}

best_summary = psample(df=era5, obs=obs, **psample_settings, **lim_dict)

# %% [markdown]
# ## Process-based calibration

# %% [markdown]
# Every parameter governs a different aspect of the water balance, and not all parameters affect every calibration variable. Therefore, we propose an **iterative process-based calibration** approach where we calibrate parameters in order of their importance using different algorithms, objective functions, and calibration variables. Details on the calibration strategy can be found in the model publication which is currently under review. The individual steps will be implemented in the notebook soon.
#
# <img src="images/calibration_strategy.png" alt="calibration" width="300">
#

# %% [markdown]
# ## Run MATILDA with calibrated parameters

# %% [markdown]
# The following parameter set was computed applying the mentioned calibration strategy on an HPC cluster.
# <a id="param"></a>

# %%
param = {
    'RFS': 0.15,
    'SFCF': 1,
    'CET': 0,
    'PCORR': 0.58,
    'lr_temp': -0.006,
    'lr_prec': 0.0015,
    'TT_diff': 0.76198,
    'TT_snow': -1.44646,
    'CFMAX_snow': 3.3677,
    'BETA': 1.0,
    'FC': 99.15976,
    'K0': 0.01,
    'K1': 0.01,
    'K2': 0.15,
    'LP': 0.998,
    'MAXBAS': 2.0,
    'PERC': 0.09232826,
    'UZL': 126.411575,
    'CFMAX_rel': 1.2556936,
    'CWH': 0.000117,
    'AG': 0.54930484
}



print('Calibrated parameter set:\n\n')
for key in param.keys(): print(key + ': ' + str(param[key]))

# %% [markdown]
# Properly calibrated, the model shows a much better results.

# %% tags=["output_scroll"]
output_matilda = matilda_simulation(era5, obs, **settings, **param)

# %% [markdown]
# In addition to the standard plots we can explore the results interactive `ploty` plots. Go ahead and zoom as you like or select/deselect individual curves.

# %%
output_matilda[9].show()

# %% [markdown]
# The same works for the long-term seasonal cycle.

# %%
output_matilda[10].show()


# %% [markdown]
# ## Reducing the parameter space - Sensitivity analysis with FAST

# %% [markdown]
# To reduce the computation time of the calibration procedure, we need to reduce the number of parameters to be optimized. Therefore, we will perform a global sensitivity analysis to identify the most important parameters and set the others to default values. The algorithm of choice will be the [Fourier Amplitude Sensitivity Test (FAST)](https://www.tandfonline.com/doi/abs/10.1080/00401706.1999.10485594) available through the [SPOTPY](https://github.com/thouska/spotpy/blob/master/src/spotpy/algorithms/fast.py) library. As before, we will show the general procedure with a few iterations, but display results from extensive runs on a HPC. You can use the results as a guide for your parameter choices, but keep in mind that they are highly correlated with your catchment properties, such as elevation range and glacier coverage.

# %% [markdown]
# First, we calculate the required number of iterations using a formula from [Henkel et. al. 2012](https://www.informs-sim.org/wsc12papers/includes/files/con308.pdf). We choose the SPOTPY default frequency step of 2 and set the interference factor to a maximum of 4, since we assume a high intercorrelation of the model parameters. The total number of parameters in MATILDA is 21.

# %%
def fast_iter(param, interf=4, freqst=2):
    """
    Calculates the number of parameter iterations needed for parameterization and sensitivity analysis using FAST.
    Parameters
    ----------
    param : int
        The number of input parameters being analyzed.
    interf : int
        The interference factor, which determines the degree of correlation between the input parameters.
    freqst : int
        The frequency step, which specifies the size of the intervals between each frequency at which the Fourier transform is calculated.
    Returns
    -------
    int
        The total number of parameter iterations needed for FAST.
    """
    return (1 + 4 * interf ** 2 * (1 + (param - 2) * freqst)) * param

print('Needed number of iterations for FAST: ' + str(fast_iter(21)))

# %% [markdown]
# That is a lot of iterations! Running this routine on a single core would take about two days, but can be sped up significantly with each additional core. The setup would look exactly like the parameter optimization before.
#
# **Note:** No matter what number of iterations you define, SPOTPY will run $N*k$ times, where $k$ is the number of model parameters. So even if we set `rep=10`, the algorithm will run at least 21 times.

# %% tags=["output_scroll"]
from matilda.mspot_glacier import psample

fast_settings = {'rep': 52437,                              # Choose wisely before running
                 'target_mb': None,
                 'algorithm': 'fast',
                 'dbname': 'fast_matilda_example',
                 'dbname': dir_output + 'fast_example',
                 'dbformat': 'csv'
                      }
psample_settings.update(fast_settings)

print('Settings for FAST:\n\n')
for key in psample_settings.keys(): print(key + ': ' + str(psample_settings[key]))

# fast_results = psample(df=era5, obs=obs, **psample_settings)

# %% [markdown]
# We ran *FAST*s for the example catchment with the full number of iterations required. The results are saved in *CSV* files. We can use the `spotpy.analyser` library to create easy-to-read data frames from the databases. The summary shows the first (`S1`) and the total order sensitivity index (`ST`) for each parameter. `S1` refers to the variance of the model output explained by the parameter, holding all other parameters constant. The `ST` takes into account the interaction of the parameters and is therefore a good measure of the impact of individual parameters on the model output.

# %%
import spotpy
import os
import contextlib

def get_si(fast_results: str, to_csv: bool = False) -> pd.DataFrame:
    """
    Computes the sensitivity indices of a given FAST simulation results file.
    Parameters
    ----------
    fast_results : str
        The path of the FAST simulation results file.
    to_csv : bool, optional
        If True, the sensitivity indices are saved to a CSV file with the same
        name as fast_results, but with '_sensitivity_indices.csv' appended to
        the end (default is False).
    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the sensitivity indices and parameter
        names.
    """
    if fast_results.endswith(".csv"):
        fast_results = fast_results[:-4]  # strip .csv
    results = spotpy.analyser.load_csv_results(fast_results)
    # Suppress prints
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        SI = spotpy.analyser.get_sensitivity_of_fast(results, print_to_console=False)
    parnames = spotpy.analyser.get_parameternames(results)
    sens = pd.DataFrame(SI)
    sens['param'] = parnames
    sens.set_index('param', inplace=True)
    if to_csv:
        sens.to_csv(os.path.basename(fast_results) + '_sensitivity_indices.csv', index=False)
    return sens

display(get_si(dir_input + 'FAST/' + 'example_fast_nolim.csv'))

# %% [markdown]
# If you have additional information on certain parameters, limiting their bounds can have a large impact on sensitivity. For our example catchment, field observations showed that the temperature lapse rate and precipitation correction were unlikely to exceed a certain range, so we limited the parameter space for both and ran a *FAST* again.

# %%
lim_dict = {'lr_temp_lo': -0.007, 'lr_temp_up': -0.005, 'PCORR_lo': 0.5, 'PCORR_up': 1.5}
# fast_results = psample(df=era5, obs=obs, **psample_settings, **lim_dict)

# %% [markdown]
# To see the effect of parameter restrictions on sensitivity, we can plot the indices of both runs. Feel free to explore further by adding more *FAST* outputs to the plot function.

# %%
import plotly.graph_objs as go
import pandas as pd
import plotly.io as pio

def plot_sensitivity_bars(*dfs, labels=None, show=False, bar_width=0.3, bar_gap=0.6):
    """
    Plots a horizontal bar chart showing the total sensitivity index for each parameter in a MATILDA model.

    Parameters
    ----------
    *dfs : pandas.DataFrame
        Multiple dataframes containing the sensitivity indices for each parameter.
    labels : list of str, optional
        Labels to use for the different steps in the sensitivity analysis. If not provided, the default
        labels will be 'No Limits', 'Step 1', 'Step 2', etc.
    bar_width : float, optional
        Width of each bar in the chart.
    bar_gap : float, optional
        Space between bars.
    """
    traces = []
    colors = ['darkblue', 'orange', 'purple', 'cyan']   # add more colors if needed
    for i, df in enumerate(dfs):
        df = get_si(df)
        if i > 0:
            if labels is None:
                label = 'Step ' + str(i)
            else:
                label = labels[i]
        else:
            label = 'No Limits'
        trace = go.Bar(y=df.index,
                       x=df['ST'],
                       name=label,
                       orientation='h',
                       marker=dict(color=colors[i]),
                       width=bar_width)
        traces.append(trace)
    layout = go.Layout(title=dict(text='<b>' +'Total Sensitivity Index for MATILDA Parameters' + '<b>', font=dict(size=24)),
                   xaxis_title='Total Sensitivity Index',
                   yaxis_title='Parameter',
                   yaxis=dict(automargin=True),
                   bargap=bar_gap,
                   height=700)
    fig = go.Figure(data=traces, layout=layout)
    if show:
        fig.show()
        
step1 = dir_input + 'FAST/' + 'example_fast_nolim.csv'
step2 = dir_input + 'FAST/' + 'era5_ipynb_fast_step1.csv'

plot_sensitivity_bars(step1, step2, labels=['No Limits', 'lr_temp and PCORR limited'], show=True)

# %% [markdown]
# From this figure, we can easily identify the most important parameters. Depending on your desired accuracy and computational resources, you can choose a sensitivity threshold. Let's say we want to fix all parameters with a sensitivity index below 0.1. For our example, this will reduce the calibration parameters to 7.

# %%
si_df = get_si(step2)
sensitive = si_df.index[si_df['ST'] > 0.10].values

print('Parameters with a sensitivity index > 0.1:')
for i in sensitive: print(i)

# %% [markdown]
# To give you an idea how much this procedure can reduce calibration time, we run the `fast_iter()` function again.

# %%
print('Needed number of iterations for FAST with all parameters: ' + str(fast_iter(21)))
print('Needed number of iterations for FAST with the 7 most sensitive parameters: ' + str(fast_iter(7)))

# %% [markdown]
# For a single core this would roughly translate into a reduction from 44h to 4h.

# %% [markdown]
# To run the calibration with the reduced parameter space, we simply need to define values for the fixed parameters and use a helper function to translate them into boundary arguments. Here we simply use the values from our last *SPOTPY* run (`param`).

# %%
from matilda.mspot_glacier import dict2bounds

fixed_param = {key: val for key, val in param.items() if key not in sensitive}
fixed_param_bounds = dict2bounds(fixed_param)
       
print('Fixed bounds:\n')
for key in fixed_param_bounds.keys(): print(key + ': ' + str(fixed_param_bounds[key]))

# %% [markdown]
# The `psample()` setup is then as simple as before.

# %% tags=["output_scroll"]
new_settings = {'rep': 10,                             # Number of model runs. For advice check the documentation of the algorithms.
                'glacier_only': False,                 # True when calibrating a entirely glacierized catchment
                'obj_dir': 'maximize',                 # should your objective funtion be maximized (e.g. NSE) or minimized (e.g. RMSE)
                'target_mb': -156,                     # Average annual glacier mass balance to target at
                'dbformat': None,                      # Write the results to a file ('csv', 'hdf5', 'ram', 'sql')
                'output': None,                        # Choose where to store the files
                'algorithm': 'lhs',                    # Choose algorithm (for parallelization: mc, lhs, fast, rope, sceua or demcz)
                'dbname': 'era5_matilda_example',      # Choose name
                       
                'parallel': False,                     # Distribute the calculation on multiple cores or not
                # 'cores': 20                           # Set number of cores when running parallel
                      }
psample_settings.update(new_settings)

best_summary = psample(df=era5, obs=obs, **psample_settings, **fixed_param_bounds)

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Note:</b> The number of iterations required depends on the selected algorithm and the number of free parameters. To choose the correct setting, consult the <a href='https://spotpy.readthedocs.io/en/latest/Algorithm_guide/'>SPOTPY documentation</a> and the original literature for each algorithm.</div>

# %% [markdown]
# Now, we save the best parameter set for use in the next notebook. The set is stored in the `best_summary` variable.

# %% [markdown]
# **Note:** In our example we will skip this step and use the [result from the calibration on an HPC cluster](#param).

# %%
# param = best_summary['best_param']

# %% [markdown]
# Finally, we store the parameter set in a `.yml` file.

# %%
write_yaml(param, dir_output + 'parameters.yml')
print(f"Parameter set stored in '{dir_output}parameters.yml'")

# %%
import shutil

# refresh `output_download.zip` with data retrieved within this notebook
shutil.make_archive('output_download', 'zip', 'output')
print('Output folder can be download now (file output_download.zip)')

# %%
# %reset -f
