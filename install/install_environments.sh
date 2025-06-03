#!/bin/bash

#
# (1) User-wide installation:
#
# 	in a conda base environment, run:
#
# 		source install_environments.sh
#
# (2) System-wide installation
#
# 		sudo -s
#
# 		# activate conda base environment
# 		source /path/to/conda/activate base
#
# 		source install_environments.sh -p /usr/local
#
# 	For more information on a system-wide installation, check:
#
# 		https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments
# 
# 		The prefix is given as /usr/local as it seems /share/jupyter/kernels
# 		is automatically appended
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

# Parse input arguments

OPTIND=0
JUPYTER_PREFIX=""

while getopts "h?p:" opt; do
    case "$opt" in
    h|\?)
        echo "Usage: [sudo] $0 [-p /path/to/jupyter/kernels]"
		return
        ;;
    p)  JUPYTER_PREFIX=$OPTARG
        ;;
    esac
done

shift "$((OPTIND-1))"

[ "${1:-}" = "--" ] && shift

# Check if CONDA_PREFIX is set

if [ -z "${CONDA_PREFIX}" ]; then
	echo "The CONDA_PREFIX variable is empty."
	echo "Possibly, the conda base environment is not activated or conda is not installed."
	echo "Exiting."
	return
fi

# Check if currently in the base environment

if [ "$CONDA_DEFAULT_ENV" != "base" ]; then
	echo "This script should be run in the base conda environment."
	echo "Exiting."
	return
fi

# Remember the prefix of the base environment

CONDA_BASE_PREFIX=$CONDA_PREFIX

# Save path to the root installation directory

INSTALL_ROOT_DIR=$PWD

# Check if installation is user-wide or system-wide

if [ -z "${JUPYTER_PREFIX}" ];

	then
		IPYKERNEL_MODE="--user"

	else
		IPYKERNEL_MODE="--prefix=$JUPYTER_PREFIX"

fi

### Install all environments and kernels

for i in 2 3 4 6 7 8 10
do

	echo "-----------------"
	echo "Installing T$i..."
	echo "-----------------"

	T_PATH="$INSTALL_ROOT_DIR/T$i"

	source $T_PATH/install.sh
	source $CONDA_BASE_PREFIX/bin/activate base

done
