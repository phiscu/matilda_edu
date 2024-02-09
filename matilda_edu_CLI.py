import warnings
warnings.filterwarnings("ignore")

import subprocess
import os

# Get the current working directory of the "master script"
master_script_dir = os.getcwd()

# Specify the path to the subfolder containing the scripts
subfolder_path = os.path.join(master_script_dir, 'CLI')

# Iterate over the scripts in the subfolder
# %%time

for script in os.listdir(subfolder_path):
    # Construct the full path to each script
    script_path = os.path.join(subfolder_path, script)
    # Run the script as a subprocess with the same working directory as the "master script"
    subprocess.call(['python', script_path], cwd=master_script_dir)


# Possible arguments:
    # NB3: resampling intv, parquet or pickle