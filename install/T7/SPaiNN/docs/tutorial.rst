=======================
SPaiNN Workflow Example
=======================
.. _tut_00_general_info:

The general workflow with **SPaiNN** is to:

#. Generate a SPaiNN database (see `Tutorial 1 <https://spainn.readthedocs.io/en/latest/tut_examples/tut_01_preparing_data.html>`_)

#. Train a NN model for predicting various properties, *e.g.* energies and forces (see `Tutorial 2 <https://spainn.readthedocs.io/en/latest/tut_examples/tut_02_MLFF.html>`_), or energies, forces and nonadiabatic couplings (see `Tutorial 3 <https://spainn.readthedocs.io/en/latest/tut_examples/tut_03_MLFF_phase_prop.html>`_)

#. Use the model, *e.g.* for predicting the properties at a certain geometry (see `Tutorial 4 <https://spainn.readthedocs.io/en/latest/tut_examples/tut_04_predictions.html>`_) or during a SHARC simulation (see `Tutorial 5 <https://spainn.readthedocs.io/en/latest/tut_examples/tut_05_SHARC_MD.html>`_)

This workflow is in the following sections demonstrated using the methylenimmonoum cation (:math:`CH_2NH_2^+`) as sample molecule.

.. nbgallery::

  tut_examples/tut_01_preparing_data
  tut_examples/tut_02_MLFF
  tut_examples/tut_03_MLFF_phase_prop
  tut_examples/tut_04_predictions
  tutorials/tut_05_SHARC_NAMD
