#!/bin/bash

#
# (1) User-wide installation:
#
#       in a conda base environment, run:
#
#               source install.sh
#
# (2) System-wide installation
#
#               sudo -s
#
#               # activate conda base environment
#               source /path/to/conda/activate base
#
#               source install.sh -p /usr/local
#
#       For more information on a system-wide installation, check:
#
#               https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments
#
#               The prefix is given as /usr/local as it seems /share/jupyter/kernels
#               is automatically appended
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

conda create -y -n $T_NO -c conda-forge -c anaconda ipykernel 

source $CONDA_BASE_PREFIX/bin/activate $T_NO
python -m ipykernel install $IPYKERNEL_MODE --name=$T_NO
python -m pip install -r $T_PATH/requirements.txt

if [ "$EUID" == 0 ]
then

        echo "Building LAMMPS..."

        # Export CUDA environment variables
        export CUDA_HOME=/usr/local/cuda
        export PATH=${CUDA_HOME}/bin:${PATH}
        export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH 

        # Install openmpi-dev
        apt-get install libopenmpi-dev

        # Remember current directory
        START_DIRECTORY=$(pwd)

        ### Start building lammps

        cd /opt

        # Clone lammps and pair_allegro
        git clone --depth=1 https://github.com/lammps/lammps
        git clone https://github.com/mir-group/pair_allegro.git --branch v0.6.0

        # Patch lammps
        cd pair_allegro && bash patch_lammps.sh ../lammps/
        cd ..
        rm -rf pair_allegro

        # Get libtorch
        wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip
        unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cu124.zip
        rm libtorch-cxx11-abi-shared-with-deps-2.5.1+cu124.zip
        LIBTORCH_PATH=$(realpath libtorch)

        # Fix libtorch
        sed -i 's/${PROJECT_SOURCE_DIR}\/third_party\/NVTX\/c\/include/\/usr\/local\/cuda-12.9\/targets\/x86_64-linux\/include/' libtorch/share/cmake/Caffe2/public/cuda.cmake

        # Build lammps
        cd lammps
        rm -rf build
        mkdir -p build
        cd build

        cmake ../cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_PATH -DMKL_INCLUDE_DIR="$CONDA_PREFIX/include" -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
        make -j4

        ### End building lammps
        cd $START_DIRECTORY

fi

source $CONDA_BASE_PREFIX/bin/activate base
