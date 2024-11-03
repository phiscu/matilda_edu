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
# - check memory and close other notebooks
#
# - download output data an restart binder
#
# - try ovh cluster if launch fails

# %% [markdown]
# If you have downloaded your progess into `output_download.zip` file, execute the cell below to unzip its contents into `output` folder to continue where you left off.

# %%
import zipfile

# Specify the path to the zip file and the directory to extract to
zip_file_path = 'output_download.zip'
extract_to_path = 'output'

# Open the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all the contents into the specified directory
    zip_ref.extractall(extract_to_path)

print(f"Extracted all files to {extract_to_path}")
