# School on Machine Learning for Molecules and Materials Research (ML4MMR)
Welcome to the DAEMON **School on Machine Learning for Molecules and Materials Research ([ML4MMR](https://www.cecam.org/workshop-details/school-on-machine-learning-for-molecules-and-materials-research-1379)) !**

The school is held in [**Zadar, Croatia**](https://maps.app.goo.gl/ghrk4jbUWV7TUb7F8) *(June 9, 2025 - June 13, 2025)*.

This repository contains the tutorials held at **ML4MMR**.

## Table of Contents
 1. [Introduction](#1-introduction)
 2. [Tutorials Schedule](#2-tutorials-schedule)
 3. [Setting Up](#3-setting-up)
    
    3.1. [Login to JupyterHub](#31-login-to-jupyterhub)
    
    3.2. [Install the Virtual Machine](#32-install-the-virtual-machine)
    
 5. [Running the Tutorials](#4-running-the-tutorials)
 6. [Installation](#5-installation)

## 1. Introduction

TODO - Write general introduction

## 2. Tutorials Schedule

| Day | Time  | Lecturer                     | Kernel / Environment |
| --- | ----- | ---------------------------  | -------------------- |
| Mon | 16:15 | Nicola Molinari              | TODO / T1            |
| Tue | 09:45 | Ilyes Batatia                | [T2](tutorials/T2)   |
| Tue | 12:00 | Jonathan Schmidt             | TODO / T3            |
| Tue | 16:15 | Tristan Bereau & Luis Walter | TODO / T4            |
| Wed | 09:45 | Martin Uhrin                 | TODO / T5            |
| Wed | 16:15 | Milica Todorović             | [T6](tutorials/T6)   |
| Thu | 09:45 | Johannes Dietschreit         | [T7](tutorials/T7)   |
| Thu | 12:00 | Lucas Foppa                  | TODO / T8            |
| Fri | 09:45 | Robert Pinsler               | TODO / T9            |
| Fri | 12:00 | Andrés M Bran                | [T10](tutorials/T10) |

## 3. Setting Up

TODO - Insert intro to setting up

### 3.1. Login to JupyterHub

TODO - Insert instructions on how to login to JupyterHub.

### 3.2. Install the Virtual Machine

TODO - Insert instructions on how to install the virtual machine.

## 4. Running the Tutorials

 1. [Login to JupyterHub](#31-login-to-jupyterhub) or start a JupyterLab instance locally/in your [Virtual Machine](#32-install-the-virtual-machine)
 2. Using the [Tutorials Schedule](#2-tutorials-schedule), identify to which Kernel / Environment the current tutorial corresponds to
 3. Locate the tutorial Jupyter Notebook in the [tutorials](tutorials) directory and open it
 4. In JupyterLab, click ``Kernel`` &rarr; ``Change Kernel`` and select the kernel for the tutorial
 5. Enjoy the tutorial!

## 5. Installation

Currently, this chapter covers only the installation of Python packages
necessary to run the tutorials. System requirements will be added as necessary.

First, clone this repository:

```bash
git clone https://github.com/rbi-mtm/ml4mmr
```

The main installation script is given in [install/install_environments.sh](install/install_environments.sh).

This script will install a separate [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) and [IPython kernel](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments) for each tutorial. 
The installed kernels may then be activated in Jupyter Notebook as described in [4. Running the Tutorials](#4-running-the-tutorials).

[install_environments.sh](install/install_environments.sh) may perform a user-wide or system-wide installation. The script must be run from the [install](install) directory and in the conda ``base`` environment.

User-wide installation:

```bash
cd ml4mmr/install
source /path/to/conda/bin/activate base
source install_environments.sh
```

System-wide installation:

```bash
cd ml4mmr/install
sudo -s
source /path/to/conda/bin/activate base
source install_environments.sh -p /usr/local
```

For more information on the system-wide installation, check the [IPython documentation](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments).

[install_environments.sh](install/install_environments.sh) invokes a separate environment-installing script for each tutorial. The scripts may be run stand-alone in an analogous way to (re)install the environment/kernel only for a given tutorial.
E.g., to install system-wide packages only for the [T2](tutorials/T2) tutorial:

```bash
cd ml4mmr/install/T2
sudo -s
source /path/to/conda/bin/activate base
source install.sh -p /usr/local
```
