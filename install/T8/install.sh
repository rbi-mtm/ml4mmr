#!/bin/bash

#
# (1) User-wide installation:
#
# 	in a conda base environment, run:
#
# 		source install.sh
#
# (2) System-wide installation
#
# 		sudo -s
#
# 		# activate conda base environment
# 		source /path/to/conda/activate base
#
# 		source install.sh -p /usr/local
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
# This script installs the kernel/environment for one tutorial
#
# It is assumed the script is run from the same directory it is located in.
#
#######
#

# Check if the script is being run from ../install_environments.sh
# or from current directory

if [ -z "${INSTALL_ROOT_DIR}" ]; then

		# Running from current directory

		T_PATH=$PWD

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

		# Check if installation is user-wide or system-wide

		if [ -z "${JUPYTER_PREFIX}" ];

			then
				IPYKERNEL_MODE="--user"

			else
				IPYKERNEL_MODE="--prefix=$JUPYTER_PREFIX"

			fi

fi

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

# Get tutorial number

T_NO=$(basename "$T_PATH")

### Install

conda create -y -n $T_NO -c conda-forge -c anaconda ipykernel -c bioconda java-jdk

source $CONDA_BASE_PREFIX/bin/activate $T_NO
python -m ipykernel install $IPYKERNEL_MODE --name=$T_NO
python -m pip install -r $T_PATH/requirements.txt

source $CONDA_BASE_PREFIX/bin/activate base
