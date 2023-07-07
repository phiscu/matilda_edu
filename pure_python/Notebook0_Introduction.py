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
# # Introduction

# %% [markdown]
# Welcome to **MATILDA**, the Python workflow for Modeling Water Resources in Glacierized Catchments! In the following series of notebooks you will go all the way from data acquisition to the analysis of climate change impacts on your catchment. Every notebook tackles an individual step in the modeling workflow. 
#
# - [Notebook 1 - Catchment Delineation](Notebook1_Catchment_deliniation.ipynb) delineates your catchment and downloads all types of static data such as the digital elevation model, glacier outlines, and ice thickness.
#
# - [Notebook 2 - Forcing data](Notebook2_Forcing_data.ipynb) downloads and processes ERA5-Land reanalysis data to calibrate the glacio-hydrological model.
#
# - [Notebook 3 - CMIP6](Notebook3_CMIP6.ipynb) downloads and processes CMIP6 climate model data for a historical period and two emission pathways until 2100.
#
# - [Notebook 4 - MATILDA](Notebook4_MATILDA.ipynb) runs a glacio-hydrological model for your catchment with default parameters and guides you through the calibration process.
#
# - [Notebook 5 - MATILDA scenarios](Notebook5_MATILDA_scenarios.ipynb) uses your calibrated parameter set so run the model for all CMIP6 ensemble members.
#
# - [Notebook 6 - Analysis](Notebook6_Analysis.ipynb) visualizes the ensemble output in interactive plots.
#
# - [Notebook 7 - Climate Change Indicators](Notebook7_Climate_Change_Impact.ipynb) calculates a set of  meteorological and hydrological indicators from your results and visualizes them in interactive figures.
#
# **Note**: *Although all notebooks can be executed in a Binder, the model calibration is a resource intensive task that will be very slow on a single CPU. You can speed up the process by downloading the Notebook and your data, and running it on a local computer with more cores. Other options to reduce calibration time are outlined in Notebook 4.*
#
# Have fun exploring!

# %% [markdown]
# ## Signing up for Google Earth Engine

# %% [markdown]
# Much of the public data acquisition will be done using the [Google Earth Engine Python API](https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api). This not only allows us to access an unique collection of public datasets but to "outsource" most of their preprocessing to Google servers. If you want to use this service, you need to sign up for an Earth Engine Account. You can do this with an existing Google Account or create a new one with any mail account.

# %% [markdown]
# 2. To start creating your account click on *Get Started* in the top right corner.
#
# ![enter image description here](https://i.postimg.cc/nzMyrSfG/start2.png)

# %% [markdown]
# 3. Log into your Google account or create one, if you don't have a Google account yet.
#
# ![enter image description here](https://i.postimg.cc/7YQqNzm6/sign-in-google.png)

# %% [markdown]
# 4. Once you signed into your Google account, you can register your first project. Click on *Register a Noncommercial or Commercial Cloud project*.
#
# ![](https://i.postimg.cc/hjqDdqM1/get-started.png)

# %% [markdown]
# 5. Next, choose how you want to use Earth Engine. Click on *Unpaid usage* and choose *Academia & Research*.
#
# ![](https://i.postimg.cc/2y0rMLY0/for-academia.png)

# %% [markdown]
# 6. Now you have the option to create a new Google Cloud Project or to choose an existing Google Cloud Project. 
# Create a new project by clicking on *Create a new Google Cloud Project*. Then you'll have to choose your organization, create a project ID and optionally choose a project name. Click on *CONTINUE TO SUMMARY*.
#
# ![enter image description here](https://i.postimg.cc/pXd28CQ4/ID3.png)

# %% [markdown]
# 7. Before your project is registered you might be asked to accept the Terms of Services if you haven't done so already. 
# Click on *Cloud Terms of Services*. You will be redirected to your Google account where you can accept the terms of services.
#
# ![enter image description here](https://i.postimg.cc/Wb91fCbs/cloud-terms.png)

# %% [markdown]
# 8. Confirm your Cloud Project information by clicking on *CONFIRM AND CONTINUE*.
# ![enter image description here](https://i.postimg.cc/L8n8dDnL/confirm.png)

# %% [markdown]
#

# %% [markdown]
#
