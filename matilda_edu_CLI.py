import warnings
warnings.filterwarnings("ignore")
import subprocess
import os

# Define the list of Python scripts to be executed
scripts = ['nb1_static_data.py']


# Iterate over the list of scripts and execute each one using subprocess.call()
for script in scripts:
    subprocess.call(['python', os.getcwd() + '/CLI/' + script])
