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
# **Note:** On a single CPU one MATILDA run over 120y takes ~4s. For all ensemble members this adds up to ~4min. The `MatildaBulkProcessor` class allows you to reduce this time significantly with more CPUs so you might want to run this notebook locally. Or have a coffee. Again...
#

# %% [markdown]
# # Change to parquet

# %%
from tools.helpers import update_yaml, read_yaml, write_yaml
import configparser

# read local config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

# get directories from config.ini
dir_input = config['FILE_SETTINGS']['DIR_INPUT']
dir_output = config['FILE_SETTINGS']['DIR_OUTPUT']


# %% [markdown]
# ## Do not change the yaml but only the dict!!

# %%
import pandas as pd
matilda_settings = read_yaml(f"{dir_output}/settings.yml")
adapted_settings = {
    "set_up_start": '1979-01-01',  # Start date of the setup period
    "set_up_end": '1980-12-31',  # End date of the setup period
    "sim_start": '1981-01-01',  # Start date of the simulation period
    "sim_end": '2100-12-31',  # End date of the simulation period
    "plots": False
}

update_yaml(f"{dir_output}/settings.yml", adapted_settings)
matilda_settings['glacier_profile'] = pd.read_csv(f"{dir_output}/glacier_profile.csv")

for key in matilda_settings.keys(): print(key + ': ' + str(matilda_settings[key]))

# %%
param_dict = read_yaml(f"{dir_output}/parameters.yml")

# %%
from tools.helpers import pickle_to_dict

## Read adjusted CMIP6 data
tas = pickle_to_dict(f"{dir_output}cmip6/adjusted/tas.pickle")
pr = pickle_to_dict(f"{dir_output}cmip6/adjusted/pr.pickle")


# %% [markdown]
# # Continue here...

# %%
def cmip2df(temp, prec, scen, col):
    """
    Converts temperature and precipitation data from a CMIP model output dictionary into a Pandas DataFrame.
    Parameters
    ----------
    temp : dict
        dictionary of temperature data from a CMIP model
    prec : dict
        dictionary of precipitation data from a CMIP model
    scen : str
        name of the scenario (e.g. RCP4.5)
    col : str
        name of the column containing data for the scenario (e.g. tas)
    Returns:
    ----------
    df : pandas.DataFrame
        DataFrame containing the temperature and precipitation data for the given scenario and column
    """
    df = pd.DataFrame({'T2': temp[scen][col], 'RRR': prec[scen][col]}).reset_index()
    df.columns = ['TIMESTAMP', 'T2', 'RRR']
    return df


# %%
## Create MATILDA input

import pandas as pd


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
dict_to_pickle(scenarios, test_dir + 'adjusted/matilda_input.pickle')

# scenarios = pickle_to_dict(test_dir + 'adjusted/matilda_input.pickle')

# %%
## Run Matilda in a loop (takes a while - have a coffee)

from tqdm import tqdm
import contextlib
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

    def run_multi_process(self):
        """
        Runs the MATILDA simulations for the scenarios and models in multi-processing mode and returns a dictionary
        of results.
        Returns
        -------
        dict
            A dictionary of MATILDA simulation results.
        """

        out_dict = {}  # Create an empty dictionary to store the outputs
        with Pool() as pool:
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
matilda_scenarios = matilda_bulk.run_multi_process()

dict_to_pickle(matilda_scenarios, test_dir + 'adjusted/matilda_scenarios.pickle')

# matilda_scenarios = pickle_to_dict(test_dir + 'adjusted/matilda_scenarios.pickle')


# %%
## Store results in parquet files to limit storage costs

import pandas as pd
import os
from fastparquet import write
from tqdm import tqdm


def dict_to_parquet(dictionary: dict, directory_path: str, pbar: bool = True) -> None:
    """
    Recursively stores the dataframes in the input dictionary as parquet files in the specified directory.
    Nested dictionaries are supported. If the specified directory does not exist, it will be created.
    Parameters
    ----------
    dictionary : dict
        A nested dictionary containing pandas dataframes.
    directory_path : str
        The directory path to store the parquet files.
    pbar : bool, optional
        A flag indicating whether to display a progress bar. Default is True.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    if pbar:
        bar_iter = tqdm(dictionary.items(), desc='Writing parquet files: ')
    else:
        bar_iter = dictionary.items()
    for k, v in bar_iter:
        if isinstance(v, dict):
            dict_to_parquet(v, os.path.join(directory_path, k), pbar=False)
        else:
            file_path = os.path.join(directory_path, k + ".parquet")
            write(file_path, v, compression='GZIP')


def parquet_to_dict(directory_path: str, pbar: bool = True) -> dict:
    """
    Recursively loads the dataframes from the parquet files in the specified directory and returns a dictionary.
    Nested directories are supported.
    Parameters
    ----------
    directory_path : str
        The directory path containing the parquet files.
    pbar : bool, optional
        A flag indicating whether to display a progress bar. Default is True.
    Returns
    -------
    dict
        A dictionary containing the loaded pandas dataframes.
    """
    dictionary = {}
    if pbar:
        bar_iter = tqdm(sorted(os.listdir(directory_path)), desc='Reading parquet files: ')
    else:
        bar_iter = sorted(os.listdir(directory_path))
    for file_name in bar_iter:
        file_path = os.path.join(directory_path, file_name)
        if os.path.isdir(file_path):
            dictionary[file_name] = parquet_to_dict(file_path, pbar=False)
        elif file_name.endswith(".parquet"):
            k = file_name[:-len(".parquet")]
            dictionary[k] = pd.read_parquet(file_path)
    return dictionary


# Store dictionary in Parquet files
dict_to_parquet(matilda_scenarios, test_dir + 'adjusted/parquet')
# Load dictionary from Parquet files
matilda_scenarios = parquet_to_dict(test_dir + 'adjusted/parquet')


# %%
## Create custom dataframes for analysis

def custom_df(dic, scenario, var, resample_freq=None):
    """
    Takes a dictionary of model outputs and returns a combined dataframe of a specific variable for a given scenario.
    Parameters
    -------
    dic : dict
        A nested dictionary of model outputs.
        The outer keys are scenario names and the inner keys are model names.
        The corresponding values are dictionaries containing two keys:
        'model_output' (DataFrame): containing model outputs for a given scenario and model
        'glacier_rescaling' (DataFrame): containing glacier properties for a given scenario and model
    scenario : str
        The name of the scenario to select from the dictionary.
    var : str
        The name of the variable to extract from the model output DataFrame.
    resample_freq : str, optional
        The frequency of the resulting time series data.
        Defaults to None (i.e. no resampling).
        If provided, should be in pandas resample frequency string format.
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the combined data of the specified variable for the selected scenario
        and models. The DataFrame is indexed by the time steps of the original models.
        The columns are the names of the models in the selected scenario.
    Raises
    -------
    ValueError
        If the provided  var  string is not one of the following: ['avg_temp_catchment', 'avg_temp_glaciers',
        'evap_off_glaciers', 'prec_off_glaciers', 'prec_on_glaciers', 'rain_off_glaciers', 'snow_off_glaciers',
        'rain_on_glaciers', 'snow_on_glaciers', 'snowpack_off_glaciers', 'soil_moisture', 'upper_groundwater',
        'lower_groundwater', 'melt_off_glaciers', 'melt_on_glaciers', 'ice_melt_on_glaciers', 'snow_melt_on_glaciers',
        'refreezing_ice', 'refreezing_snow', 'total_refreezing', 'SMB', 'actual_evaporation', 'total_precipitation',
        'total_melt', 'runoff_without_glaciers', 'runoff_from_glaciers', 'total_runoff', 'glacier_area',
        'glacier_elev', 'smb_water_year', 'smb_scaled', 'smb_scaled_capped', 'smb_scaled_capped_cum', 'surplus']
    """
    out1_cols = ['avg_temp_catchment', 'avg_temp_glaciers', 'evap_off_glaciers',
                 'prec_off_glaciers', 'prec_on_glaciers', 'rain_off_glaciers',
                 'snow_off_glaciers', 'rain_on_glaciers', 'snow_on_glaciers',
                 'snowpack_off_glaciers', 'soil_moisture', 'upper_groundwater',
                 'lower_groundwater', 'melt_off_glaciers', 'melt_on_glaciers',
                 'ice_melt_on_glaciers', 'snow_melt_on_glaciers', 'refreezing_ice',
                 'refreezing_snow', 'total_refreezing', 'SMB', 'actual_evaporation',
                 'total_precipitation', 'total_melt', 'runoff_without_glaciers',
                 'runoff_from_glaciers', 'total_runoff']

    out2_cols = ['glacier_area', 'glacier_elev', 'smb_water_year',
                 'smb_scaled', 'smb_scaled_capped', 'smb_scaled_capped_cum',
                 'surplus']

    if var in out1_cols:
        output_df = 'model_output'
    elif var in out2_cols:
        output_df = 'glacier_rescaling'
    else:
        raise ValueError("var needs to be one of the following strings: " +
                         str([out1_cols, out2_cols]))

    # Create an empty list to store the dataframes
    dfs = []
    # Loop over the models in the selected scenario
    for model in dic[scenario].keys():
        # Get the dataframe for the current model
        df = dic[scenario][model][output_df]
        # Append the dataframe to the list of dataframes
        dfs.append(df[var])
    # Concatenate the dataframes into a single dataframe
    combined_df = pd.concat(dfs, axis=1)
    # Set the column names of the combined dataframe to the model names
    combined_df.columns = dic[scenario].keys()
    # Resample time series
    if resample_freq is not None:
        if output_df == 'glacier_rescaling':
            if var in ['glacier_area', 'glacier_elev']:
                combined_df = combined_df.resample(resample_freq).mean()
            else:
                combined_df = combined_df.resample(resample_freq).sum()
        else:
            if var in ['avg_temp_catchment', 'avg_temp_glaciers']:
                combined_df = combined_df.resample(resample_freq).mean()
            else:
                combined_df = combined_df.resample(resample_freq).sum()

    return combined_df

custom_df(matilda_scenarios, scenario='SSP5', var='smb_water_year', resample_freq='Y')

# %%
