# MATILDA-Edu: Workflow for Modeling Water Resources in Glacierized Catchments
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phiscu/matilda_edu/main?labpath=Notebook0_Introduction.ipynb)

Welcome to **MATILDA-Edu**, the online companion to the **MATILDA** glacio-hydrological modeling framework. This repository hosts the extended MATILDA workflow in form of a Jupyter Book. Designed for researchers, practitioners, and students, this workflow guides users from data acquisition to the analysis of climate change impacts on glacierized catchments.

📚 **Explore the Jupyter Book** on the [MATILDA-Online Website](https://matilda-online.github.io/jbook).

---

## Installation

You can run most of the workflow in an online environment hosted in mybinder.org. However, calibrating the model is computationally intensive and will be slow to run on a single CPU. For a comprehensive calibration, we recommend downloading the notebooks and running them on a local machine with multi-core processing capabilities.

To run the MATILDA-Edu workflow locally, follow these steps:

1. Clone this repository to your local machine and navigate to its root folder:
```
git clone https://github.com/phiscu/matilda_edu.git
cd matilda_edu
```
2. Create and activate a Python environment using the provided environment.yml file. We recommend the use of conda:
```
conda env create -f environment.yml
conda activate matilda_edu
```
3. Install Jupyter Notebook or Jupyter Lab if not already installed:
```
conda -c conda-forge install jupyterlab
```
4. Launch the Jupyter Notebook interface:
```
jupyter lab
```
---

## Workflow Overview

The MATILDA-Edu workflow is organized into a series of interactive Jupyter notebooks. These cover all key steps of modeling water resources in glacierized catchments, including catchment delineation, data acquisition, model calibration, and scenario analysis. Below is a detailed flowchart of the workflow:

![Workflow Flowchart](images/workflow_detailed_2024_-Full+legend.png)

---

## Core Routines

The core routines of MATILDA, including the temperature-index melt model and HBV hydrological model, are maintained in the [MATILDA repository](https://github.com/cryotools/matilda). The MATILDA-Edu workflow integrates these routines into a streamlined educational framework.

---

## Authors

- **Alexander Georgi** ([GitHub](https://github.com/geoalxx))
- **Phillip Schuster** ([GitHub](https://github.com/phiscu))
- **Mia Janzen** ([Github](https://github.com/hoepke))

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.




