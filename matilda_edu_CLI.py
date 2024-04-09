import warnings
warnings.filterwarnings("ignore")
import subprocess
import os
import json

## Arguments:

steps = ['nb5']       # ['all'] or any of ['nb1', 'nb2', 'nb3', 'nb4', 'nb5']

add_kwargs = {
    'nb1_static_data.py': {},
    'nb2_era5l.py': {},
    'nb3_cmip6.py': {},
    'nb4_matilda_calibration.py': {},
    'nb5_matilda_ensembles.py': {'num_cores': 4}
}

# MATILDA parameter set
# param = {'lr_temp': -0.006472598,
#          'lr_prec': 0.00010296448,
#          'BETA': 4.625306,
#          'CET': 0.2875196,
#          'FC': 364.81818,
#          'K0': 0.28723368,
#          'K1': 0.015692418,
#          'K2': 0.004580627,
#          'LP': 0.587188,
#          'MAXBAS': 6.730105,
#          'PERC': 1.1140852,
#          'UZL': 198.82584,
#          'PCORR': 0.74768984,
#          'TT_snow': -1.3534238,
#          'TT_diff': 0.70977557,
#          'CFMAX_ice': 2.782649,
#          'CFMAX_rel': 1.2481626,
#          'SFCF': 0.879982,
#          'CWH': 0.0020890352,
#          'AG': 0.8640329,
#          'RFS': 0.21825151}

param = {'lr_temp': -0.0057516084, 'lr_prec': 0.0015256472, 'BETA': 5.6014814, 'FC': 323.61023, 'K0': 0.124523245, 'K1': 0.01791149, 'K2': 0.006872296, 'LP': 0.5467752, 'MAXBAS': 5.325173, 'PERC': 2.9256027, 'UZL': 354.0794, 'TT_snow': 0.5702063, 'TT_diff': 1.9629607, 'CFMAX_ice': 5.2739882, 'CFMAX_rel': 1.2821848, 'CWH': 0.05004947, 'AG': 0.5625456, 'RFS': 0.2245709, 'PCORR': 0.64, 'SFCF': 1, 'CET': 0}

# Dump parameter set
param_string = json.dumps(param)

## Run sub-scripts
# Get the current working directory of the "master script"
master_script_dir = os.getcwd()

# Specify the path to the subfolder containing the scripts
subfolder_path = os.path.join(master_script_dir, 'CLI')
step_list = sorted(os.listdir(subfolder_path))

# Run all scripts or only selected ones
if steps == ['all']:
    script_list = step_list
else:
    script_list = [item for item in step_list if any(step in item for step in steps)]

# Iterate over the scripts in the subfolder

for script in script_list:
    # Construct the full path to each script
    script_path = os.path.join(subfolder_path, script)
    # Set an environment variable to identify as subprocess
    env = os.environ.copy()
    env["MATILDA_PARENT"] = "true"
    # Pass parameters and additional arguments to corresponding scripts
    if script == 'nb4_matilda_calibration.py':
        env["MATILDA_PARAMS"] = param_string
    # Check if additional variables are defined for this script
    if script in add_kwargs:
        for var_name, var_value in add_kwargs[script].items():
            env[var_name] = str(var_value)
    subprocess.call(['python', script_path], env=env, cwd=master_script_dir)




# IT DOESN'T WORK TO PASS VARIABLES TO THE SUB-SCRIPTS THAT WAY...





# Possible arguments:
    # NB3: resampling intv, parquet or pickle
    # NB4: parameter set from file,
    # NB5: parquet or pickle


# Probleme:

