#!/bin/bash

#
# Usage:
#
# in a conda base environment, run:
#
# source install_T2.sh
#

# Check if CONDA_PREFIX is set

if [ -z "${CONDA_PREFIX}" ]; then
	echo "The CONDA_PREFIX variable is empty."
	echo "Possibly, the conda base environment is not activated or conda is not installed."
	echo "Exiting."
	exit 1
fi

# Check if currently in the base environment

if [ "$CONDA_DEFAULT_ENV" != "base" ]; then
	echo "This script should be run in the base conda environment."
	echo "Exiting."
	exit 1
fi

# Remember the prefix of the base environment

CONDA_BASE_PREFIX=$CONDA_PREFIX

# Check if the script is being run from ../../../install_environments.sh
# or from the same directory it is located in

if [ -z "${TUTORIALS}" ]; then

	T_PATH=$PWD

fi

### Install

conda create -y -n T2 -c conda-forge -c anaconda ipykernel xtb-python

source $CONDA_BASE_PREFIX/bin/activate T2
python -m ipykernel install --user --name=T2
python -m pip install -r $T_PATH/requirements_T2.txt

source $CONDA_BASE_PREFIX/bin/activate base
