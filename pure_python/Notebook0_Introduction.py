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
# # Introduction

# %% [markdown]
# Welcome to **MATILDA-Online**, the Python-based workflow for **Modeling Water Resources in Glacierized Catchments**! This book describes the comprehensive toolkit in detail and guides you step-by-step from data acquisition to analysis of climate change impacts on the selected catchment. Designed with flexibility and accessibility in mind, MATILDA integrates robust scientific models, public data sources, and user-friendly tools to make sophisticated glacio-hydrological modeling accessible to researchers, practitioners, and students alike.
#
# The workflow is divided into a series of interactive notebooks, each focused on a specific component of the modeling process. These notebooks streamline complex tasks such as catchment delineation, data processing, model calibration, and climate scenario analysis, ensuring clarity and reproducibility at each step:
#
# - **[Notebook 1 - Catchment Delineation](Notebook1_Catchment_delineation.ipynb):** Delineate your catchment and retrieve static geospatial data, including digital elevation models, glacier outlines, and ice thickness distributions.
#   
# - **[Notebook 2 - Forcing Data](Notebook2_Forcing_data.ipynb):** Acquire and process ERA5-Land reanalysis data, preparing inputs for glacio-hydrological model calibration.
#
# - **[Notebook 3 - CMIP6 Climate Data](Notebook3_CMIP6.ipynb):** Download and process historical and future climate data from the Coupled Model Intercomparison Project Phase 6 (CMIP6) for two emission scenarios.
#
# - **[Notebook 4 - MATILDA Model](Notebook4_MATILDA.ipynb):** Run the MATILDA model with default parameters and calibrate it based on mutiple objectives.
#
# - **[Notebook 5 - Scenario Simulations](Notebook5_MATILDA_scenarios.ipynb):** Apply your calibrated parameter set to run the model over all CMIP6 ensemble members for robust scenario-based analysis.
#
# - **[Notebook 6 - Result Analysis & Impact Assessment](Notebook6_Analysis.ipynb):** Visualize model output in interactive plots across ensemble simulations, extract key meteorological and hydrological indicators of of climate change impacts, and create a visual summary.
#
#
# The workflow below is demonstrated using a sample site in the Tian Shan Mountains of Kyrgyzstan. To try the toolkit for yourself, simply click on the rocket icon in the toolbar above to launch an online environment hosted by [mybinder.org](https://mybinder.org/). There you can run any notebook with the sample data or upload your own and edit the config file accordingly. Note that while most of the workflow will work fine in the binder, calibrating the model is computationally intensive and will be slow to run on a single CPU. For a comprehensive calibration that takes full advantage of the [spotpy](https://spotpy.readthedocs.io/en/latest/) library, we recommend downloading the notebooks and running them on a local machine with multi-core processing capabilities. Additional options to reduce calibration time are described in Notebook 4.
#
# Have fun exploring and happy modeling!
#
# ![flowchart](images/workflow_detailed_2024_-Full_legend.png)

# %% [markdown]
# ## Signing up for Google Earth Engine (GEE)

# %% [markdown]
# Much of the public data acquisition will be done using the [Google Earth Engine Python API](https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api). This not only allows us to access an unique collection of public datasets but to "outsource" most of their preprocessing to Google servers. Therefore, you require an Earth Engine Account to use this service. If you don't have one, sign up as follows.

# %% [markdown]
# 1. To start visit the [Earth Engine website](https://earthengine.google.com/) and click on *Get Started* in the top right corner.
#
# ![enter image description here](https://i.postimg.cc/nzMyrSfG/start2.png)

# %% [markdown]
# 2. Log into your Google account or create one using any email adress.
#
# ![enter image description here](https://i.postimg.cc/7YQqNzm6/sign-in-google.png)

# %% [markdown]
# 3. Once you signed in you can register your first project. Click on *Register a Noncommercial or Commercial Cloud project*.
#
# ![](https://i.postimg.cc/hjqDdqM1/get-started.png)

# %% [markdown]
# 4. Next, choose how you want to use Earth Engine. You may select *Unpaid usage* and *Academia & Research*.
#
# ![](https://i.postimg.cc/2y0rMLY0/for-academia.png)

# %% [markdown]
# <a id="step-5">5. </a>Now you have the option to join an existing **Google Cloud Project** or create a new one.
# For the latter click on *Create a new Google Cloud Project*, and choose your organization, create a project ID and optionally choose a project name. Click on *CONTINUE TO SUMMARY* when finished.
#
#
#
# ![enter image description here](https://i.postimg.cc/pXd28CQ4/ID3.png)

# %% [markdown]
# 6. Before your project is registered you might be asked to accept the Terms of Services if you haven't done so already. 
# Click on *Cloud Terms of Services*. You will be redirected to your Google account where you can accept the terms.
#
# ![enter image description here](https://i.postimg.cc/Wb91fCbs/cloud-terms.png)

# %% [markdown]
# 7. Finally, confirm your Cloud Project information by clicking on *CONFIRM AND CONTINUE*.
#
# ![enter image description here](https://i.postimg.cc/L8n8dDnL/confirm.png)

# %% [markdown]
# 8. The first cell of every notebook using GEE will check your authentication status. If it is the first time the GEE API is initialized, a hyperlink will be generated that brings you to a GEE log in page. There you need to ...

# %% [markdown] pycharm={"name": "#%% md\n"}
# ## Authorize access for Google Earth Engine

# %% [markdown] pycharm={"name": "#%% md\n"}
# 1. Choose your account and project and click on *GENERATE TOKEN.*
#
# ![enter image description here](images/nb0_gee_token_1.png)

# %% [markdown] pycharm={"name": "#%% md\n"}
# 2. If not done already, you will need to sign in to your Google Account. You'll get a security notification for unverified apps. Click *continue*.
#
# ![enter image description here](https://i.postimg.cc/8PzQmGk8/continue.png)  

# %% [markdown] pycharm={"name": "#%% md\n"}
# 3. Next, grant your Earth Engine Notebook Client access to your account and click *Continue*.
#
# ![enter image description here](images/nb0_gee_token_3.png)

# %% [markdown]
# 4. Finally, copy the authorisation code ...
#  
# ![enter image description here](images/nb0_gee_token_4.png)

# %% [markdown] pycharm={"name": "#%% md\n"}
# 5. ... and paste it into the designated field in the notebook.
#
# ![enter image description here](https://i.postimg.cc/ZnfQM9bG/enter-code.png)

# %% [markdown] pycharm={"name": "#%% md\n"}
# 6. You should get a message saying *Successfully saved authorization token.* You are now ready to start with the MATILDA workflow. Before we dive into data handling, let's have a look at ...

# %% [markdown]
# ## The *config.ini* file

# %% [markdown]
# This file contains a list of essential information for the workflow and allows customization. If you want to try MATILDA-Online with the sample dataset, you only need to edit the entry ```CLOUD_PROJECT``` and change it to your projects name from <a href="#step-5">Step 5</a>.  If you want to use your own data, replace the file with the discharge observation in the ```input/``` folder and adapt the reference coordinates accordingly.

# %% [markdown]
# 1. The first section ```[FILE_SETTINGS]``` allows you to **edit paths and file names for in- and outputs**. This can especially be useful if you model multiple catchments in the same copy of the repository.
#
# 2. In the ```[CONFIG]``` section you can ...
#    - ... specify your **Google Cloud project**. This information is **mandatory** to use the GEE in the workflow. The current project ```matilda-edu``` is set up for demonstration purposes and is not publicly accessible.
#    - ... specify your **reference coordinates** (usually your gauging station location) and select the calibration period. The latter should cover your observation period plus a few years before as a spin-off.
#    - ... change the **digital elevation model** used.
#    - ... choose download option from GEE (direct download or via ```xarray```).
#    - ... choose whether to create scenario-based **projections** or just model the past.
#    - ... disable the generation of **live maps**.
#    - ... configure the **style of output figures**. More information about the available styles can be found in the **[SciencePlots manual](https://github.com/garrettj403/SciencePlots/wiki/Gallery)**.
#    - ... choose between a faster (```.pickle```) and a more compact (```.parquet```) format for intermediate files.
#    - ... set the number of cores available for computation. If you are in a binder, leave this at 1.
#    - ... decide whether you want to store your output folder in a `.zip` file at the end of every Notebook. This is useful when you work online and want to download your (intermediate) results.
#
# &nbsp;
# 3. The last section ```[MEDIA_SERVER]``` holds credentials for the file access on a file repository of our university and should not be edited if you are not a university member and know what you're doing. The credentials only grant read access to glacier-related public data and are not of value to you.

# %% [markdown]
# With the ```config.ini``` set up, you may now start with **[Notebook 1](Notebook1_Catchment_delineation.ipynb)**.
