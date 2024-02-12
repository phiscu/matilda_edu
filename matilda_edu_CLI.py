import warnings
warnings.filterwarnings("ignore")
import subprocess
import os
import json

# Get the current working directory of the "master script"
master_script_dir = os.getcwd()

# Specify the path to the subfolder containing the scripts
subfolder_path = os.path.join(master_script_dir, 'CLI')

# MATILDA parameter set
param = {'lr_temp': -0.006472598,
         'lr_prec': 0.00010296448,
         'BETA': 4.625306,
         'CET': 0.2875196,
         'FC': 364.81818,
         'K0': 0.28723368,
         'K1': 0.015692418,
         'K2': 0.004580627,
         'LP': 0.587188,
         'MAXBAS': 6.730105,
         'PERC': 1.1140852,
         'UZL': 198.82584,
         'PCORR': 0.74768984,
         'TT_snow': -1.3534238,
         'TT_diff': 0.70977557,
         'CFMAX_ice': 2.782649,
         'CFMAX_rel': 1.2481626,
         'SFCF': 0.879982,
         'CWH': 0.0020890352,
         'AG': 0.8640329,
         'RFS': 0.21825151}

# Dump parameter set
param_string = json.dumps(param)

# Iterate over the scripts in the subfolder

for script in sorted(os.listdir(subfolder_path)):
    # Construct the full path to each script
    script_path = os.path.join(subfolder_path, script)
    # Set an environment variable to identify as subprocess
    env = os.environ.copy()
    env["MATILDA_PARENT"] = "true"
    # Pass parameters to Matilda, run other scripts without arguments
    if script == 'nb4_matilda_calibration.py':
        env["MATILDA_PARAMS"] = param_string
        subprocess.call(['python', script_path], env=env, cwd=master_script_dir)
    else:
        subprocess.call(['python', script_path], env=env, cwd=master_script_dir)


# Possible arguments:
    # NB3: resampling intv, parquet or pickle
    # NB4: parameter set from file,


# Probleme:
    # - identische CMIP6 figures unterschiedlich benannt

