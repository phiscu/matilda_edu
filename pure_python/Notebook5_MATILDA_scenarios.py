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

# %% [markdown]
# # MATILDA Scenarios

# %% [markdown]
# After calibrating MATILDA we can now use the best parameter set to run the model with climate scenario data until 2100. In this notebook we will only
#
# - ...run MATILDA with the same parameters and settings but 2 x 31 different climate forcings.
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

print(f"Input path: '{dir_input}'")
print(f"Output path: '{dir_output}'")


# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Note:</b> We provide two storage options: <code>pickle</code> files are fast to read and write, but take up more disk space. You can use them on your local machine. <code>parquet</code> files are half the size but take longer to read and write. They should be your choice in the Binder.</div>

# %% [markdown]
# To run MATILDA for a period in the future, we need to adapt the modeling period. Therefore, we read the `settings.yaml` to a ditionary and change the respective settings. We also turn off the plotting module to reduce processing time and add the glacier profile from its `.csv`.

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
# As we want to use the best calibrated parameter set for the projections we read the `parameters.yml`...

# %%
param_dict = read_yaml(f"{dir_output}/parameters.yml")

# %% [markdown]
# ...and our forcing data.
#
# <div class="alert alert-block alert-info">
# <b>Note:</b> Choose either <code>pickle</code> or <code>parquet</code> depending on what you used in Notebook 3.</div>

# %%
from tools.helpers import parquet_to_dict, pickle_to_dict

# For size:
# tas = parquet_to_dict(f"{dir_output}cmip6/adjusted/tas_parquet")
# pr = parquet_to_dict(f"{dir_output}cmip6/adjusted/pr_parquet")

## For speed
tas = pickle_to_dict(f"{dir_output}cmip6/adjusted/tas.pickle")
pr = pickle_to_dict(f"{dir_output}cmip6/adjusted/pr.pickle")

# %% [markdown]
# Now we have to convert the individual climate projections into MATILDA input dataframes with the correct column names. We store these 2 x 31 MATILDA inputs in a nested dictionary again and save the file in a `parquet` (or `pickle`).

# %%
from tools.helpers import dict_to_parquet, dict_to_pickle

def create_scenario_dict(tas: dict, pr: dict, scenario_nums: list) -> dict:
    """
    Create a nested dictionary of scenarios and models from two dictionaries of pandas DataFrames.
    Parameters
    ----------
    tas : dict
        A dictionary of pandas DataFrames where the keys are scenario names and each DataFrame has columns
        representing different climate model mean daily temperature (K) time series.
    pr : dict
        A dictionary of pandas DataFrames where the keys are scenario names and each DataFrame has columns
        representing different climate models mean daily precipitation (mm/day) time series.
    scenario_nums : list
        A list of integers representing the scenario numbers to include in the resulting dictionary.
    Returns
    -------
    dict
        A nested dictionary where the top-level keys are scenario names (e.g. 'SSP2', 'SSP5') and the values are
        dictionaries containing climate models as keys and the corresponding pandas DataFrames as values.
        The DataFrames have three columns: 'TIMESTAMP', 'T2', and 'RRR', where 'TIMESTAMP'
        represents the time step, 'T2' represents the mean daily temperature (K), and 'RRR' represents the mean
        daily precipitation (mm/day).
    """
    scenarios = {}
    for s in scenario_nums:
        s = 'SSP' + str(s)
        scenarios[s] = {}
        for m in tas[s].columns:
            model = pd.DataFrame({'T2': tas[s][m],
                                  'RRR': pr[s][m]})
            model = model.reset_index()
            mod_dict = {m: model.rename(columns={'time': 'TIMESTAMP'})}
            scenarios[s].update(mod_dict)
    return scenarios

scenarios = create_scenario_dict(tas, pr, [2, 5])

print("Storing MATILDA scenario input dataframes on disk...")
# dict_to_parquet(scenarios, f"{dir_output}cmip6/adjusted/matilda_scenario_input_parquet")

dict_to_pickle(scenarios, f"{dir_output}cmip6/adjusted/matilda_scenario_input.pickle")

# %% [markdown]
# ## Running MATILDA for all climate projections

# %% [markdown]
# Now that we are set up we need to **run MATILDA for every CMIP6 model and both scenarios**. This adds up to **62 model runs at ~4s each** on a single core. So you can either start the bulk processor and have a break or download data and notebook to run it on more cores on a local computer.
#
# <div class="alert alert-block alert-info">
# <b>Note:</b> Don't be confused by the status bar. It only updates after one full scenario is processed.</div>

# %%
## Run Matilda in a loop (takes a while - have a coffee)

from matilda.core import matilda_simulation
from tqdm import tqdm
import contextlib
import os
from multiprocessing import Pool
from functools import partial


class MatildaBulkProcessor:
    """
    A class to run multiple MATILDA simulations for different input scenarios and models in single or multi-processing
    mode and store the results in a dictionary.
    Attributes
    ----------
    scenarios : dict
        A dictionary with scenario names as keys and a dictionary of climate models as values.
    matilda_settings : dict
        A dictionary of MATILDA settings.
    matilda_parameters : dict
        A dictionary of MATILDA parameter values.
    Methods
    -------
    run_single_process():
        Runs the MATILDA simulations for the scenarios and models in single-processing mode and returns a dictionary
        of results.
    run_multi_process():
        Runs the MATILDA simulations for the scenarios and models in multi-processing mode and returns a dictionary
        of results.
    matilda_headless(df, matilda_settings, matilda_parameters):
        A helper function to run a single MATILDA simulation given a dataframe, MATILDA settings and parameter
        values.
    """

    def __init__(self, scenarios, matilda_settings, matilda_parameters):
        """
        Parameters
        ----------
        scenarios : dict
            A dictionary with scenario names as keys and a dictionary of models as values.
        matilda_settings : dict
            A dictionary of MATILDA settings.
        matilda_parameters : dict
            A dictionary of MATILDA parameter values.
        """

        self.scenarios = scenarios
        self.matilda_settings = matilda_settings
        self.matilda_parameters = matilda_parameters

    @staticmethod
    def matilda_headless(df, matilda_settings, matilda_parameters):
        """
        A helper function to run a single MATILDA simulation given a dataframe, MATILDA settings and parameter
        values.
        Parameters
        ----------
        df : pandas.DataFrame
            The input dataframe for the MATILDA simulation.
        matilda_settings : dict
            A dictionary of MATILDA settings.
        matilda_parameters : dict
            A dictionary of MATILDA parameter values.
        Returns
        -------
        dict
            A dictionary containing the MATILDA model output and glacier rescaling factor.
        """

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                output = matilda_simulation(df, **matilda_settings, parameter_set=matilda_parameters)
        return {'model_output': output[0], 'glacier_rescaling': output[5]}

    def run_single_process(self):
        """
        Runs the MATILDA simulations for the scenarios and models in single-processing mode and returns a dictionary
        of results.
        Returns
        -------
        dict
            A dictionary of MATILDA simulation results.
        """

        out_dict = {}  # Create an empty dictionary to store the outputs
        # Loop over the scenarios with progress bar
        for scenario in self.scenarios.keys():
            model_dict = {}  # Create an empty dictionary to store the model outputs
            # Loop over the models with progress bar
            for model in tqdm(self.scenarios[scenario].keys(), desc=scenario):
                # Get the dataframe for the current scenario and model
                df = self.scenarios[scenario][model]
                # Run the model simulation and get the output while suppressing prints
                model_output = self.matilda_headless(df, self.matilda_settings, self.matilda_parameters)
                # Store the list of output in the model dictionary
                model_dict[model] = model_output
            # Store the model dictionary in the scenario dictionary
            out_dict[scenario] = model_dict
        return out_dict

    def run_multi_process(self, num_cores=2):
        """
        Runs the MATILDA simulations for the scenarios and models in multi-processing mode and returns a dictionary
        of results.
        Returns
        -------
        dict
            A dictionary of MATILDA simulation results.
        """

        out_dict = {}  # Create an empty dictionary to store the outputs
        with Pool(num_cores) as pool:
            # Loop over the scenarios with progress bar
            for scenario in tqdm(self.scenarios.keys(), desc="Scenarios SSP2 and SSP5"):
                model_dict = {}  # Create an empty dictionary to store the model outputs
                # Loop over the models with progress bar
                model_list = [self.scenarios[scenario][m] for m in self.scenarios[scenario].keys()]
                for model, model_output in zip(self.scenarios[scenario], pool.map(
                        partial(self.matilda_headless, matilda_settings=self.matilda_settings,
                                matilda_parameters=self.matilda_parameters), model_list)):
                    model_dict[model] = model_output
                # Store the model dictionary in the scenario dictionary
                out_dict[scenario] = model_dict
            pool.close()

        return out_dict


matilda_bulk = MatildaBulkProcessor(scenarios, matilda_settings, param_dict)
# matilda_scenarios = matilda_bulk.run_single_process()
matilda_scenarios = matilda_bulk.run_multi_process(num_cores=4)

print("Storing MATILDA scenario outputs on disk...")
# dict_to_parquet(matilda_scenarios, f"{dir_output}cmip6/adjusted/matilda_scenarios_parquet")

dict_to_pickle(matilda_scenarios, test_dir + 'adjusted/matilda_scenarios.pickle')


# %% [markdown]
# The results is a large nested dictionary with 62 x 2 dataframes of MATILDA outputs. To have a look at the results, continue with Notebook 6.
