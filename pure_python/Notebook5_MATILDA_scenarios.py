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
# # MATILDA Scenarios

# %% [markdown]
# After calibrating MATILDA we can now use the best parameter set to run the model with climate scenario data until 2100. In this notebook we will...
#
# - ...run MATILDA with the same parameters and settings for two emission scenarios and all models of the ensemble.
#
# <div class="alert alert-block alert-info">
# <b>Note:</b> On a single CPU one MATILDA run over 120y takes ~4s. For all ensemble members this adds up to ~4min. The <code>MatildaBulkProcessor</code> class allows you to reduce this time significantly with more CPUs so you might want to run this notebook locally. Or have a coffee. Again...</div>
#

# %% [markdown]
# ## Set up the scenario runs

# %% [markdown]
# As before, we start by reading our paths from the `config.ini`.

# %%
import configparser

# read local config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

# get directories from config.ini
dir_input = config['FILE_SETTINGS']['DIR_INPUT']
dir_output = config['FILE_SETTINGS']['DIR_OUTPUT']
zip_output = config['CONFIG']['ZIP_OUTPUT']

# set the file format for storage
compact_files = config.getboolean('CONFIG','COMPACT_FILES')

# get the number of available cores
num_cores = int(config['CONFIG']['NUM_CORES'])

print(f"Input path: '{dir_input}'")
print(f"Output path: '{dir_output}'")


# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Note:</b> Choose in the config between faster <code>pickle</code> files and smaller <code>parquet</code> files.</div>

# %% [markdown]
# Let's extend the modeling period to the full century. Therefore, we read the `settings.yaml` to a ditionary and change the respective settings. We also turn off the plotting module to reduce processing time and add the glacier profile from its `.csv`.

# %%
from tools.helpers import read_yaml, write_yaml
import pandas as pd
matilda_settings = read_yaml(f"{dir_output}/settings.yml")
adapted_settings = {
    "set_up_start": '1998-01-01',  # Start date of the setup period
    "set_up_end": '1999-12-31',  # End date of the setup period
    "sim_start": '2000-01-01',  # Start date of the simulation period
    "sim_end": '2100-12-31',  # End date of the simulation period
    "plots": False
}
matilda_settings['glacier_profile'] = pd.read_csv(f"{dir_output}/glacier_profile.csv")

matilda_settings.update(adapted_settings)

print("Settings for MATILDA scenario runs:\n")
for key in matilda_settings.keys(): print(key + ': ' + str(matilda_settings[key]))

# %% [markdown]
# Now, we read the calibrated parameter set from the `parameters.yml` and our forcing data from the binary files.

# %%
from tools.helpers import parquet_to_dict, pickle_to_dict

param_dict = read_yaml(f"{dir_output}/parameters.yml")
print("Parameters loaded.")

if compact_files:
    # For size:
    tas = parquet_to_dict(f"{dir_output}cmip6/adjusted/tas_parquet")
    pr = parquet_to_dict(f"{dir_output}cmip6/adjusted/pr_parquet")
else:
    # For speed
    tas = pickle_to_dict(f"{dir_output}cmip6/adjusted/tas.pickle")
    pr = pickle_to_dict(f"{dir_output}cmip6/adjusted/pr.pickle")

print("Forcing data loaded.")

# %% [markdown]
# The `create_scenario_dict` function converts the individual climate projections into MATILDA input dataframes. We store the ensemble of MATILDA inputs in a nested dictionary again and save the file in a `parquet` (or `pickle`). 

# %%
from tools.helpers import dict_to_parquet, dict_to_pickle, create_scenario_dict

scenarios = create_scenario_dict(tas, pr, [2, 5])

print("Storing MATILDA scenario input dataframes on disk...")

if compact_files:
    dict_to_parquet(scenarios, f"{dir_output}cmip6/adjusted/matilda_scenario_input_parquet")
else:
    dict_to_pickle(scenarios, f"{dir_output}cmip6/adjusted/matilda_scenario_input.pickle")


# %% [markdown]
# ## Running MATILDA for all climate projections

# %% [markdown]
# Now that we are set up we need to **run MATILDA for every CMIP6 model and both scenarios**. This adds up to **50-70 model runs at ~4s each** on a single core, depending on how many models remained in your ensemble. So you can either start the bulk processor and have a break or download the repo and change the config according to your available cores.
#
# <div class="alert alert-block alert-info">
# <b>Note:</b> Don't be confused by the status bar. It only updates after one full scenario is processed.</div>

# %%
from tools.helpers import MatildaBulkProcessor
import shutil

# Create an instance of the MatildaBulkProcessor class
matilda_bulk = MatildaBulkProcessor(scenarios, matilda_settings, param_dict)

# Run Matilda in a loop (takes a while - have a coffee)
if num_cores == 1:
    matilda_scenarios = matilda_bulk.run_single_process()
else:
    matilda_scenarios = matilda_bulk.run_multi_process(num_cores=num_cores)

print("Storing MATILDA scenario outputs on disk...")

if compact_files:
    dict_to_parquet(matilda_scenarios, f"{dir_output}cmip6/adjusted/matilda_scenarios_parquet")
else:
    dict_to_pickle(matilda_scenarios, f"{dir_output}cmip6/adjusted/matilda_scenarios.pickle")

if zip_output:
    # refresh `output_download.zip` with data retrieved within this notebook
    shutil.make_archive('output_download', 'zip', 'output')
    print('Output folder can be download now (file output_download.zip)')


# %%
# %reset -f

# %% [markdown]
# The result is a large nested dictionary with 100-140 dataframes of MATILDA outputs. Now, it is finally time to look at the results. Explore your projections in [Notebook 6](Notebook6_Analysis.ipynb).
