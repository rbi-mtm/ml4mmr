====================================
Tutorial 5 – SHARC MD with NN Model
====================================
.. _tut_05:

In `Tutorial 3 <https://spainn.readthedocs.io/en/latest/tut_examples/tut_03_MLFF_phase_prop.html>`_, we showed how to train a NN model based on energies (:math:`E_i`), forces (:math:`-\nabla_{\mathrm{R}}E_i`) and smoothed nonadiabatic couplings (:math:`\mathbf{C}_{ij}^s`).
Here we demonstrate, how to use the trained model, when running surface hopping molecular dynamics simulations with `SHARC 3.0 <https://www.sharc-md.org/>`_.

Surface hopping is a semi-classical approach used for investigating molecular dynamics involving several electronic states, distinguishing between adiabatic and nonadiabatic processes. 
In this method, the classical treatment is employed to handle the adiabatic motion of atomic nuclei. 
Meanwhile, the influence of nonadiabatic factors arising from the distribution of electronic population across different energy states is incorporated using a random-based algorithm. 
This results in trajectories that combine classical and quantum characteristics, enabling transitions between electronic states. 
To capture the probabilistic behavior of wave packet propagation, multiple of these trajectories are generated as an ensemble.

SHARC Trek: Trajectory Setup
-----------------------------
.. _05_prep_traj:

The first step for running nonadiabatic molecular dynamics (NAMD) simulations with `SHARC <https://www.sharc-md.org/>`_ is to set up trajectories. 
This involves a series of interconnected steps, which are briefly outline in the following (*cf.* the `SHARC tutorial <https://sharc-md.org/wp-content/uploads/2023/04/SHARC_Tutorial.pdf>`_ for detailed instructions).

#. **Molecular Structure Optimization:** Optimize the geometry of the molecule of interest and perform a frequency calcultaion to make sure that the optimized geometry represents a minimum geometry on the ground state potential energy surface, *i.e.*, an energetically favorable confirguration in the ground state.

#. **Wigner Sampling of Initial Conditions:** Utilize the frequency output for Wigner sampling, which generates a comprehensive collection of initial conditions (`initconds`), *i.e.* an ensemble of geometries centered around the ground state minimum geometry.

#. **Quantum Chemistry:** Compute the energies, forces, and couplings pertaining for the generated set of initial conditions, thereby establishing a comprehensive understanding of their electronic properties alongside the structural properties.

#. **Selection of Excitation-Ready Initial Conditions:** Identify the ensemble of initial conditions (`initconds.excited`) that are amenable to excitation at specific excitation wavelengths.

#. **Setup Trajectories:** Orchestrating trajectories for selected electronic states and geometries residing within `initconds.excited`. These trajectories should be initiated from each electronic state of interest, enabling an exploration of the system's behavior across various electronic configurations. 

In the last step, trajectory folders containing SHARC input files are created.
The typical parameters used in the respective input files are summarized in the following table.

.. _table_input_SHARC:

.. list-table:: Table 1: Common input parameters for classical `SHARC <https://www.sharc-md.org/>`_ simulations, as well as for SHARC simulations that utilize electronic properties (such as energies, forces, and couplings) obtained through a neural network model – specifically, *via* machine learning predictions with **SPaiNN**.
   :widths: 25 35 35
   :header-rows: 1

   * - Parameter
     - SHARC
     - SPaiNN: SHARC + SchNetPack
   * - veloc
     - external
     - external
   * - nstates
     - 3 0 0
     - 3 0 0
   * - acstates
     - 3 0 0 
     - 3 0 0
   * - state
     - 3 mch
     - 3 mch
   * - coeff
     - auto 
     - auto
   * - ezero
     - -94.7095344465
     - -94.7095344465
   * - tmax
     - 150.0
     - 150.0
   * - stepsize
     - 0.5
     - 0.5
   * - nsubsteps
     - 25
     - 25
   * - surf
     - diagonal
     - diagnoal
   * - **coupling**
     - **overlap**
     - **nacdr**
   * - gradcorrect
     - True
     - True
   * - ekincorrect
     - parallel_nac
     - parallel_nac
   * - reflect_frustrated
     - none
     - none
   * - decoherence_scheme
     - edc
     - edc
   * - decoherence_param
     - 0.1
     - 0.1
   * - hopping_procedure
     - sharc
     - sharc
   * - grad_all
     - True
     - True
   * - nac_all
     - True
     - True
   * - nospinorbit
     - True
     - True

**Note:** When using the ML model for predicting electronic properties, the input parameter `coupling` needs to be changed from overlap to nacdr, indicating that the couplings are directly obtained (from the ML model). 

Fueling SHARC with NN Predictions
----------------------------------
.. _05_interface:

Once the trajectory folders, with the SHARC `input` files suitable for **SPaiNN** (see Table 1) and all other input files (*e.g.*, `geom`, `veloc`, `QM.in`, and `run.sh`) are set up, one needs to perform the NAMD simulations.
The following steps demonstrate how to use a trained NN model, namely `best_model_E_F_C` in the NAMD simulations for the methylene immonium cation (:math:`CH_2NH_2^+`, `atoms='CNHHHH'`) starting in the electronic state :math:`S_2`, involving in total three singlet states, namely :math:`S_0`, :math:`S_1`, and :math:`S_2`.

.. code-block:: python
 
  >>> import sys, os
  >>> import spainn
  >>> from spainn.interface.sharcinterface import SHARC_NN

Note: If you get the following warning,

.. code-block:: python

  WARNING:spainn.interface:PySHARC not installed! Install PySHARC to use all features of SPaiNN

try to source the sharcvars.sh, *e.g.*, by running

.. code-block:: console

  (venv)$ source $SHARC/sharcvars.sh

.. code-block:: python

  >>> # Give multiple models for, e.g., active learning
  >>> models = ["train/best_model_E_F_C"]
  >>> th = None # for active learning, e.g., {'energy': 0.004}
  >>> nn = SHARC_NN(modelpath=models, # path to NN models
  >>>    atoms="CNHHHH", # symbols of sample molecule
  >>>    n_states={'n_singlets': 3, 'n_triplets': 0}, # dict of state numbers
  >>>    thresholds=th,
  >>>    cutoff=10.0, # for building representation
  >>>    nac_key="smooth_nacs", # model trained on smoothed nacs
  >>>    properties=['energy','forces','smooth_nacs'] # properties predicted by NN
  >>> )
  >>> nn.run_sharc("./input",0)

Decoding SHARC's Findings
--------------------------

**ToDo: Plot with populations over time... how to get there...**

