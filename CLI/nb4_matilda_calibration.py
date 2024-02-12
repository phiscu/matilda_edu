import json
import configparser
import ast
import pandas as pd
import spotpy
import contextlib
import plotly.graph_objs as go
from IPython.core.display_functions import display
import plotly.io as pio
import sys
import os
# Add change cwd to matilda_edu home dir and add it to PATH
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.append(cwd)            # Hotfix. Master script runs subprocess that changes the CWD.
sys.path.append(parent_dir)
from tools.helpers import update_yaml, read_yaml, write_yaml, drop_keys
from matilda.core import matilda_simulation
from matilda.mspot_glacier import psample, dict2bounds

## Convert passed parameter set back into a dictionary
if "MATILDA_PARENT" in os.environ:         # Check if script is running as a subprocess
    # Retrieve parameter string from environment variable
    param_string = os.environ.get("MATILDA_PARAMS")
    # Convert back to dictionary
    param = json.loads(param_string)

## Model Setup
# read local config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

# get output dir and date range from config.ini
dir_input = config['FILE_SETTINGS']['DIR_INPUT']
dir_output = config['FILE_SETTINGS']['DIR_OUTPUT']
date_range = ast.literal_eval(config['CONFIG']['DATE_RANGE'])

print('MATILDA will be calibrated on the period ' + date_range[0] + ' to ' + date_range[1])

# define setup period
length_of_setup_period = 2

sim_start = pd.to_datetime(date_range[0]) + pd.DateOffset(years=length_of_setup_period)
set_up_end = sim_start - pd.DateOffset(days=1)
dates = {'set_up_start': date_range[0],
         'set_up_end': str(set_up_end).split(' ')[0],  # remove hh:mm:ss
         'sim_start': str(sim_start).split(' ')[0],  # remove hh:mm:ss
         'sim_end': date_range[1]}

for key in dates.keys(): print(key + ': ' + dates[key])


# load settings.yml and add optional settings
update_yaml(dir_output + 'settings.yml', dates)
remaining_settings = {"freq": "M",  # aggregation level of model outputs (D, M, Y)
                      "warn": False,  # show warnings of subpackages?
                      "plot_type": "all",  # interactive and/or non-interactive plots ('print', 'interactive', 'all')
                      "elev_rescaling": True}  # treat mean glacier elevation as constant or change with glacier evolution
update_yaml(dir_output + 'settings.yml', remaining_settings)

settings = read_yaml(dir_output + 'settings.yml')
glacier_profile = pd.read_csv(dir_output + 'glacier_profile.csv')
settings['glacier_profile'] = glacier_profile

print('MATILDA settings:\n\n')
for key in settings.keys(): print(key + ': ' + str(settings[key]))


## Load forcing data

# load ERA5L and obs data
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

# Add glacier mass balance data
mass_balances = pd.read_csv(dir_input + '/hma_mb_20190215_0815_rmse.csv', usecols=['RGIId', 'mb_mwea', 'mb_mwea_sigma'])
ids = pd.read_csv(dir_output + '/RGI/Glaciers_in_catchment.csv')

merged = pd.merge(mass_balances, ids, on='RGIId')
mean_mb = round(merged['mb_mwea'].mean() * 1000, 3)  # Mean catchment MB in mm w.e.
mean_sigma = round(merged['mb_mwea_sigma'].mean() * abs(mean_mb), 3)  # Mean uncertainty of catchment MB in mm w.e.

target_mb = [mean_mb - mean_sigma, mean_mb + mean_sigma]

print('Target glacier mass balance for calibration: ' + str(mean_mb) + ' +-' + str(mean_sigma) + 'mm w.e.')


## CALIBRATION          --> ADD INTERFACE FOR MSPOT_CIRRUS ROUTINE!

# psample_settings = drop_keys(settings, ['warn', 'plots', 'plot_type'])
#
# additional_settings = {'rep': 10,  # Number of model runs. For advice check the documentation of the algorithms.
#                        'glacier_only': False,  # True when calibrating a entirely glacierized catchment
#                        'obj_dir': 'maximize',
#                        # should your objective funtion be maximized (e.g. NSE) or minimized (e.g. RMSE)
#                        'target_mb': mean_mb,  # Average annual glacier mass balance to target at
#                        'dbformat': None,  # Write the results to a file ('csv', 'hdf5', 'ram', 'sql')
#                        'output': None,  # Choose where to store the files
#                        'algorithm': 'lhs',
#                        # Choose algorithm (for parallelization: mc, lhs, fast, rope, sceua or demcz)
#                        'dbname': 'era5_matilda_example',  # Choose name
#
#                        'parallel': False,  # Distribute the calculation on multiple cores or not
#                        # 'cores': 20                           # Set number of cores when running parallel
#                        }
# psample_settings.update(additional_settings)
#
# print('Settings for calibration runs:\n\n')
# for key in psample_settings.keys(): print(key + ': ' + str(psample_settings[key]))
#
# # set parameter bounds
# lim_dict = {'lr_temp_lo': -0.007, 'lr_temp_up': -0.005, 'PCORR_lo': 0.5, 'PCORR_up': 1.5}
#
# best_summary = psample(df=era5, obs=obs, **psample_settings, **lim_dict)


## Run MATILDA with calibrated parameters
if "MATILDA_PARENT" in os.environ:         # Check if script is running as a subprocess
    print('Calibrated parameter set:\n\n')
    for key in param.keys(): print(key + ': ' + str(param[key]))
    output_matilda = matilda_simulation(era5, obs, **settings, parameter_set=param)
    # Write parameter set to .yml
    write_yaml(param, dir_output + 'parameters.yml')
    print(f"Parameter set stored in '{dir_output}parameters.yml'")
else:
    output_matilda = matilda_simulation(era5, obs, **settings)



