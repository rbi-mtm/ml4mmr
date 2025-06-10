# School on Machine Learning for Molecules and Materials Research (ML4MMR)
Welcome to the DAEMON **School on Machine Learning for Molecules and Materials Research ([ML4MMR](https://www.cecam.org/workshop-details/school-on-machine-learning-for-molecules-and-materials-research-1379)) !**

The school is held in [**Zadar, Croatia**](https://maps.app.goo.gl/ghrk4jbUWV7TUb7F8) *(June 9, 2025 - June 13, 2025)*.

This repository contains the tutorials held at **ML4MMR**.

## Table of Contents
 1. [Introduction](#1-introduction)
 2. [Tutorials Schedule](#2-tutorials-schedule)
 3. [Setting Up](#3-setting-up)
    
    3.1. [Login to JupyterHub](#31-login-to-jupyterhub)
    
    3.2. [Local Installation](#32-local-installation)
    
 4. [Running the Tutorials](#4-running-the-tutorials)
 5. [Installation](#5-installation)
 6. [Common Issues](#6-common-issues)

## 1. Introduction

First, read the chapter on [Setting Up](#3-setting-up). Once done, proceed to
the instructions on [Running the Tutorials](#4-running-the-tutorials). Some
frequently encountered issues are listed in [Common Issues](#6-common-issues).

> [!TIP]
> Some participants might not be able to [login](#31-login-to-jupyterhub)
> or [run the tutorials](#4-running-the-tutorials) due
> to a technical issue.
>
> Therefore, we encourage you to work in groups.
> Be collegial and share both knowledge and technical resources!

## 2. Tutorials Schedule

| Day | Time  | Lecturer                     | Kernel / Environment | Colab Link |
| --- | ----- | ---------------------------  | -------------------- | ---------- |
| Mon | 16:15 | Nicola Molinari              | [T1](tutorials/T1)   | [![badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1f9rGNDaLqKVFhAK4mTpu-5tn-BvXEmNs?usp=sharing)
| Tue | 09:45 | Ilyes Batatia                | [T2](tutorials/T2)   | [![badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZrTuTvavXiCxTFyjBV4GqlARxgFwYAtX) [![badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oCSVfMhWrqHTeHbKgUSQN9hTKxLzoNyb)  |
| Tue | 12:00 | Jonathan Schmidt             | [T3](tutorials/T3)   | |
| Tue | 16:15 | Tristan Bereau & Luis Walter | [T4](tutorials/T4)   | [![badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jcS-q8DnS6O4n4BLS4yvldE5yMbOFDcY?usp=sharing) |
| Wed | 09:45 | Martin Uhrin                 | [T5](tutorials/T5)   | [![badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Q_qqTultR3T7YisoNnTBXWBm8reXQ6JL?usp=sharing) [![badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XX5Qx47CqnGIFC0IJxMSw_uH-N14Je1m?usp=sharing) |
| Wed | 16:15 | Milica Todorović             | [T6](tutorials/T6)   | [![badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17Vy4iTuIyvOh37pbuI4OeO1YTUmpm-hz#sandboxMode=true)  [![badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18SR3N0N9HaVXNu76F-wXk8Y5ZJ1N0q_b?usp=sharing) |
| Thu | 09:45 | Johannes Dietschreit         | [T7](tutorials/T7)   | |
| Thu | 12:00 | Lucas Foppa                  | [T8](tutorials/T8)   | |
| Fri | 09:45 | Robert Pinsler               | T9                   | |
| Fri | 12:00 | Andrés M Bran                | [T10](tutorials/T10) | |

> [!NOTE]
> The Google Colab links are given for some of the tutorials only for convenience and redundancy.
> They are **not** tested and might not be updated.

## 3. Setting Up

> [!TIP]
> If you encounter any issues while setting up, please send a message through the school Slack workspace
or to [juraj.ovcar@gmail.com](mailto:juraj.ovcar@gmail.com).

***
### 3.1. Login to JupyterHub

> [!Important]
> You should have received an invitation link from the domain [srce.hr](srce.hr).
>
> If you have not received the invitation or encounter issues with logging in
> through [eduGAIN](https://edugain.org/), immediately contact the organizers.

After accepting the invitation:

 1. Open [https://jupyter.srce.hr](https://jupyter.srce.hr) in your browser
 2. Click on ``Prijava putem eduGAIN/AAI@EduHr`` and login
 3. Select the ``School on Machine Learning for Molecules and Materials Research`` server option and click ``Start``
 4. If successful, proceed to [Running the Tutorials](4-running-the-tutorials)

***
### 3.2. Local Installation

If you wish to install the softtware and run the tutorials on your own machine, read the chapter on [Installation](5-installation).

## 4. Running the Tutorials

> [!IMPORTANT]
> When logging in for the first time, you must first download the tutorials.
>
> In the Launcher tab, scroll down and click on ``Terminal``. Type in the following commands:
>
> ```bash
> git-lfs install
> git clone https://github.com/rbi-mtm/ml4mmr
> ```

 1. Using the [Tutorials Schedule](#2-tutorials-schedule), identify to which Kernel / Environment the current tutorial corresponds to
 2. Locate the tutorial Jupyter Notebook in the [tutorials](tutorials) directory and open it
 3. In JupyterLab, click ``Kernel`` &rarr; ``Change Kernel`` and select the kernel for the tutorial
 4. Enjoy the tutorial!

## 5. Installation

Currently, this chapter covers only the installation of Python packages
necessary to run the tutorials. System requirements will be added as necessary.

First, clone this repository:

```bash
git-lfs install
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

## 6. Common Issues

> [!WARNING]
>
> The most common issue is getting a ``ModuleNotFound`` error while running a tutorial.
>
> Check if you selected the correct [Jupyter kernel](4-running-the-tutorials). If you still encounter ``ModuleNotFound``, report this to the organizers.
>
> If the issue persists after selecting the correct kernel, a possible fix is to open up a ``Terminal`` and run the following (replace T3 with the tutorial you have an issue with):
>
> ```bash
> source /opt/conda/bin/activate base
> conda activate T3
> pip install missing_module
> ```
>
> After the ``pip`` installation, in JupyterLab, click on ``Kernel`` &rarr; ``Restart Kernel``.

***

Other common issues:

> I started my server on [https://jupyter.srce.hr](https://jupyter.srce.hr) but it takes very long to start up. I'm even seeing a ``Timeout`` error.

- The server is downloading an updated container image. Wait for 5-10 minutes and try again.


> I'm running ``git clone https://github.com/rbi-mtm/ml4mmr`` and it's stuck at ``Updating...``

- You forgot to run ``git-lfs install``. Press ``Ctrl+C`` and run:

```bash
rm -rf ml4mmr
git-lfs install
git clone https://github.com/rbi-mtm/ml4mmr
```
