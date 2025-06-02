
Welcome to SpaiNN's documentation!
===================================

**spaiNN** is a Python package that provides a flexible and efficient interface to the `SchNetPack 2.0 <https://github.com/atomistic-machine-learning/schnetpack/tree/master>`_ package a toolbox for the development and application of deep neural networks to the prediction of potential energy surfaces and other quantum-chemical properties of molecules and materials.
spaiNN allows users to predict energies, forces, dipoles, and non-adiabatic couplings for multiple electronic states, and additionally provides an interface to the `SHARC <https://www.sharc-md.org/>`_ (Surface Hopping including Arbitrary Couplings) software for running excited-state dynamics simulations.
spaiNN is an extension to the `SchNarc <https://github.com/schnarc/SchNarc>`_ [1] software, *i.e.*, a python software that combines `SchNetPack 1.0 <https://github.com/atomistic-machine-learning/schnetpack/tree/schnetpack1.0>`_ [2-4] and `SHARC <https://www.sharc-md.org/>`_.

It offers a *simple* and *intuitive* python and command line API.

Features
----------

- Predict potential energy surfaces of multiple electronic states (SchNet [1-4])
- Predict vector-properties of multiple electronic states, such as non-adiabatic couplings or dipole moments (SchNet [1-4], PaiNN [5])
- Interface to the `SHARC <https://www.sharc-md.org/>`_ software for running excited state dynamics simulations
- Flexible implementation in Python 

Check out the `User Guide <userguide/overview.rst>`_ section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development.

Contents
----------

.. toctree::
   :glob:
   :caption: Get Started
   :maxdepth: 1

   installation
   namd_properties

.. toctree::
   :glob:
   :caption: User Guide
   :maxdepth: 1

   userguide/overview
   userguide/data_pipeline
   userguide/models   
   userguide/md

.. toctree::
   :glob:
   :caption: Tutorials
   :maxdepth: 1

   tutorial
   
.. toctree::
   :glob:
   :caption: SPaiNN
   :maxdepth: 1

   api/properties
   api/multidatamodule
   api/asetools
   api/model
   api/loss
   api/metric
   api/calculator
   api/plotting
   api/interface
   api/cli

References
------------

- [1] J. Westermayr, M. Gastegger, P. Marquetand, *Phys. Chem. Lett.* **2020**, 11, 10, 3828–3834, `10.1021/acs.jpclett.0c00527 <https://doi.org/10.1021/acs.jpclett.0c00527>`_
- [2] K.T. Schütt. F. Arbabzadah. S. Chmiela, K.-R. Müller, A. Tkatchenko, *Nat. Comm.* **2017**, 8, 13890, `10.1038/ncomms13890 <https://www.nature.com/articles/ncomms13890>`_
- [3] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller, *Advances in Neural Information Processing Systems* **2017**, 30, 992-1002, `Paper <https://proceedings.neurips.cc/paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html>`_
- [4] K.T. Schütt. P.-J. Kindermans, H. E. Sauceda, S. Chmiela, A. Tkatchenko, K.-R. Müller, *J. Chem. Phys.* **2018**, 148, 24, 241722, `10.1063/1.5019779 <https://aip.scitation.org/doi/10.1063/1.5019779>`_
- [5] K. T. Schütt, O. T. Unke, M. Gastegger, *Proceedings of the 38th International Conference on Machine Learning* **2021**, PMLR 139:9377-9388, `Paper <https://proceedings.mlr.press/v139/schutt21a.html>`_
