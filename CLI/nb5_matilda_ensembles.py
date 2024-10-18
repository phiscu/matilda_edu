import configparser
import pandas as pd
from tqdm import tqdm
import contextlib
from multiprocessing import Pool
from functools import partial
from matilda.core import matilda_simulation
import os
import sys
# Add change cwd to matilda_edu home dir and add it to PATH
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.append(cwd)            # Hotfix. Master script runs subprocess that changes the CWD.
sys.path.append(parent_dir)
from tools.helpers import read_yaml, write_yaml, parquet_to_dict, pickle_to_dict, dict_to_parquet, dict_to_pickle

## Parsed arguments

num_cores = 5

## Setup
# read local config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

# get directories from config.ini
dir_input = config['FILE_SETTINGS']['DIR_INPUT']
dir_output = config['FILE_SETTINGS']['DIR_OUTPUT']

print(f"Input path: '{dir_input}'")
print(f"Output path: '{dir_output}'")

# Read and adapt settings
matilda_settings = read_yaml(f"{dir_output}/settings.yml")
adapted_settings = {
    "set_up_start": '1979-01-01',  # Start date of the setup period
    "set_up_end": '1980-12-31',  # End date of the setup period
    "sim_start": '1981-01-01',  # Start date of the simulation period
    "sim_end": '2100-12-31',  # End date of the simulation period
    "plots": False
}
matilda_settings['glacier_profile'] = pd.read_csv(f"{dir_output}/glacier_profile.csv")
matilda_settings.update(adapted_settings)
print("Settings for MATILDA scenario runs:\n")
for key in matilda_settings.keys(): print(key + ': ' + str(matilda_settings[key]))

# Read parameters and forcing data
param_dict = read_yaml(f"{dir_output}/parameters.yml")

# For size:
# tas = parquet_to_dict(f"{dir_output}cmip6/adjusted/tas_parquet")
# pr = parquet_to_dict(f"{dir_output}cmip6/adjusted/pr_parquet")

# For speed
tas = pickle_to_dict(f"{dir_output}cmip6/adjusted/tas.pickle")
pr = pickle_to_dict(f"{dir_output}cmip6/adjusted/pr.pickle")

## Arrange all models and scenarios in a nested dictionary

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
dict_to_pickle(scenarios, f"{dir_output}cmip6/adjusted/matilda_scenario_input.pickle")
# dict_to_parquet(scenarios, f"{dir_output}cmip6/adjusted/matilda_scenario_input_parquet")

## Running MATILDA for all climate projections

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
if num_cores == 1:
    print("Running MATILDA in single-processing mode.")
    matilda_scenarios = matilda_bulk.run_single_process()
elif num_cores < 1:
    raise ValueError("Number of cores must be at least 1")
else:
    print(f"Running MATILDA in multi-processing mode on {num_cores} cores.")
    matilda_scenarios = matilda_bulk.run_multi_process(num_cores=num_cores)

print("Storing MATILDA scenario outputs on disk...")
# dict_to_parquet(matilda_scenarios, f"{dir_output}cmip6/adjusted/matilda_scenarios_parquet")
dict_to_pickle(matilda_scenarios, f"{dir_output}cmip6/adjusted/matilda_scenarios.pickle")
print('Done.')

scenarios_test = pickle_to_dict(f"{dir_output}cmip6/adjusted/matilda_scenarios.pickle")