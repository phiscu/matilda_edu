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
# # Running MATILDA for the training period

# %% [markdown]
# - create settings.yaml (here or in the other Notebooks)
# - run MATILDA with default parameters
# - (split in calibration and validation samples)
# - run mspot with few iterations
# - write best parameter set to yaml
# - run matilda with best parameter set

# %% [markdown]
# Some helper functions to work with yaml files

# %%
import yaml

def read_yaml(file_path):
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
        return data
    
def write_yaml(data, file_path):
    with open(file_path, 'w') as f:
        yaml.safe_dump(data, f)

def update_yaml(file_path, new_items):
    data = read_yaml(file_path)
    data.update(new_items)
    write_yaml(data, file_path)



# %% [markdown]
# Read data required for MATILDA from the config file

# %%
import configparser
import ast

# read local config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

# get output dir and date range from config.ini
dir_output = config['FILE_SETTINGS']['DIR_OUTPUT']
date_range = ast.literal_eval(config['CONFIG']['DATE_RANGE'])

# %% [markdown]
# Derive setup and modeling periods from the defined time period. Default is to use the first two years as spinup.

# %%
import pandas as pd

length_of_setup_period = 2

sim_start = pd.to_datetime(date_range[0]) + pd.DateOffset(years = length_of_setup_period)
set_up_end = sim_start - pd.DateOffset(days = 1)

dates = {'set_up_start': date_range[0],
        'set_up_end': str(set_up_end).split(' ')[0],        # remove hh:mm:ss
        'sim_start': str(sim_start).split(' ')[0],          # remove hh:mm:ss
        'sim_end': date_range[1]}

print(dates)

# %% [markdown]
# Append the dates to the settings.yml with stored catchment information.

# %%
update_yaml(dir_output + 'settings.yml', dates)

# %% [markdown]
# Check the remaining settings, append them to the settings file and load it as a dictionary.

# %%
remaining_settings = {"freq": "M",            # aggregation level of model outputs (D, M, Y)
                      "warn": False,          # show warnings of subpackages?
                      "plot_type": "all",     # interactive and/or non-interactive plots ('print', 'interactive', 'all')
                      "elev_rescaling": True  # 
                     }

update_yaml(dir_output + 'settings.yml', remaining_settings)

settings = read_yaml(dir_output + 'settings.yml')
glacier_profile = pd.read_csv(dir_output + 'glacier_profile.csv')
settings['glacier_profile'] = glacier_profile
print(settings)

# %% [markdown]
# # Run MATILDA with default parameters

# %% [markdown]
# Load forcing and obs data

# %%
era5 = pd.read_csv(dir_output + 'ERA5L.csv', usecols=['temp', 'prec', 'dt'])
era5.columns = ['T2', 'RRR', 'TIMESTAMP']

obs = pd.read_csv('input/' + 'obs_runoff_example.csv')

# %%
from matilda.core import matilda_simulation

output_matilda = matilda_simulation(era5, obs, **settings)

# %% [markdown]
# This is obviously an incorrect result so the model requires calibration.

# %%
param_dict = param = {'lr_temp': -0.006715786655857773,
 'lr_prec': 0.0009426868309736729,
 'BETA': 4.755073554352201,
 'CET': 0.07412818445635777,
 'FC': 424.03083598449393,
 'K0': 0.24661844658100276,
 'K1': 0.013814926672937655,
 'K2': 0.01877384431953609,
 'LP': 0.7699373762379815,
 'MAXBAS': 2.911911446589711,
 'PERC': 1.7425269942489015,
 'UZL': 392.21464659707215,
 'PCORR': 0.796841923720716,
 'TT_snow': -0.46045194130701805,
 'TT_diff': 1.7514302424196948,
 'CFMAX_ice': 7.7265119371929885,
 'CFMAX_rel': 1.6284621286938152,
 'SFCF': 0.989796885705358,
 'CWH': 0.17529112240136024,
 'AG': 0.5942337539192579,
 'RFS': 0.14722479457349263}

# %%
output_matilda = matilda_simulation(era5, obs, **settings, parameter_set = param_dict)

# %%
output_matilda[9].show()

# %% [markdown]
# # Calibrate MATILDA using the SPOTPY library

# %%
from matilda.mspot_glacier import psample

# %% [markdown]
# The default parameter boundaries of the `mspot()` function uses can be found in the MATILDA documentation. If you want to narrow down the parameter space you can do that using the following syntax.

# %%
lr_temp_lo = -0.007; lr_temp_up = -0.005

PCORR_lo_era = 0.29; PCORR_up_era = 1.2

lim_dict = {'lr_temp_lo': lr_temp_lo, 'lr_temp_up': lr_temp_up, 'PCORR_lo': PCORR_lo_era, 'PCORR_up': PCORR_up_era}

#best_summary = psample(df=era5, obs=obs, rep=10, #output=output_path + '/glacier_only',
                                    **dates, freq="D",
                                     area_cat=295.51935296803777, area_glac=31.81370047643339, lat=42.33,
                                     ele_dat=3335.668840874115, ele_cat=3293.491688025922, ele_glac=4001.8798828125,
                                     glacier_profile=glacier_profile, elev_rescaling=True,
                                     glacier_only=False,
                                     obj_dir="maximize",
                                    **lim_dict,
                                     #target_mb=-156,
                                     parallel=False, dbformat=None, algorithm='lhs', #cores=20,
                                     dbname='era5_matilda_edu_test')

# %% [markdown]
# # Run MATILDA with calibrated parameters

# %% [markdown]
# Pass best parameter set from calibration runs as dictionary.

# %%
# Test edits
