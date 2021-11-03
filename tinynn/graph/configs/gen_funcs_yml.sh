#!/bin/bash

set -e

declare -a vers=("110" "19" "18" "17")

source /data/miniconda3/etc/profile.d/conda.sh

for ver in "${vers[@]}"
do
   echo "Activating environment torch$ver"
   conda activate "torch$ver"
   which python3
   python3 gen_funcs_yml.py
done
