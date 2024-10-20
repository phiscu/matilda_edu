#!/bin/bash

# This script will find all .ipynb files and clear their outputs.
find . -name "*.ipynb" -exec jupyter nbconvert --clear-output --inplace {} +
