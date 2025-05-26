#!/bin/bash

#
# Usage:
#
# in a conda base environment, run:
#
# source install_environments.sh
# 
######
#
# The aim of this script is to automatize the installation process of
# the conda environments/kernels necessary to run the tutorials
# at ML4MMR.
#
# Currently, the script is experimental.
# Hopefully, it will be possible to install _all_ the environments
# using this script.
#
# It is assumed the script is run from the same directory it is located in.
#
#######
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

# Save path to the root installation directory

INSTALL_ROOT_DIR=$PWD

### Install all environments and kernels

for i in 2
do

	echo "----------------"
	echo "Installing T$i..."
	echo "----------------"

	T_PATH="$INSTALL_ROOT_DIR/T$i"

	source $T_PATH/install.sh
	source $CONDA_BASE_PREFIX/bin/activate base

done
