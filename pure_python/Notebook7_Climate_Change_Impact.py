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
# # Climate Change Impact Analysis

# %% [markdown]
# To highlight the impacts of climate change on our catchment we can calculate a set of indicators frequently used in climate impact studies. In this notebook we will...
#
# 1. ...create **custom data frames** containing individual output variables from all ensemble members,
#
# 2. ...**plot the ensemble mean** of these variables with a 90% confidence interval,
#
# 3. ...and create an **interactive application** to explore the results.

# %%

# %%
from tools.helpers import pickle_to_dict, parquet_to_dict
import configparser

# read output directory from config.ini file
config = configparser.ConfigParser()
config.read('config.ini')
dir_output = config['FILE_SETTINGS']['DIR_OUTPUT']

matilda_scenarios =

matilda_scenarios = pickle_to_dict(test_dir + 'adjusted/matilda_scenarios.pickle')   # pickle for speed/parquet for size


# %%

# %%

# %%
