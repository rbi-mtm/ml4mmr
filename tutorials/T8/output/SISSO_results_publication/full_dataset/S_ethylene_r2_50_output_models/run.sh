#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./Raven.out.%j
#SBATCH -e ./Raven.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J SISSO
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=72
#
#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.mpg.de
#
# Wall clock limit (max. is 24 hours):
#SBATCH --time=12:00:00

# Load compiler and MPI modules (must be the same as used for compiling the code)
###  Max Cores per node of Raven = 72  ###

module purge

ulimit -s unlimited
#module load anaconda/3/2023.03 gcc/10 openmpi/4.1  
module load anaconda/3/2023.03 intel/2024 mkl openmpi/4.1
conda activate sisso_intel
#export LD_LIBRARY_PATH=$I_MPI_ROOT/intel64//lib/:$I_MPI_ROOT/intel64//lib/release/:$INTEL_HOME/compilers_and_libraries_2020/linux/lib/intel64/:$LD_LIBRARY_PATH


srun /u/lfoppa/sissopp_param/bin/sisso++ sisso.json > sisso.out




