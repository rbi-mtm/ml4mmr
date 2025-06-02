==============
Data pipeline
==============
.. _data_pipeline:

The data pipeline of **SpaiNN** mainly consists of :py:class:`DatabaseUtils` and involves the creation of a database using the :py:class:`GenerateDB` class or the conversion of existing databases (*e.g.*, SchNarc database) using the :py:class:`ConvertDB` class. 

Generate a SPaiNN Database
----------------------------

After running a SHARC job, a couple of folders similar to the example below are created.

..  code-block:: console
    
    sharc_project  
    │
    └───ICOND_00000
    │   │   QM.in
    │   │   QM.out
    |   |   ...
    │   
    └───ICOND_00001
    │   │   QM.in
    │   │   QM.out
    |   |   ...
    │   │
    ...


When generating a database, the following properties are parsed from SHARC output, *i.e.*, ``QM.in`` and ``QM.out`` files: energies (:math:`\mathsf{E_{i}}`), forces (:math:`\mathsf{\mathbf{F}_{i}}`), non-adiabatic coupling (:math:`\mathsf{\mathbf{NAC}_{ij}}`), Dipoles (:math:`\mathsf{\mu_{i}}`), and spin-orbit couplings (:math:`\mathsf{\mathbf{SOC}_{ij}}`).
These properties obtain valuable information about the molecules, which are employed in non-adiabatic molecular dynamics simulations using SHARC (see :ref:`main`).
The respective properties are stored in form of an Atomic Simulation Environment (ASE) databse.

By default, a dictionary of units of all properties and the atomic coordinates is added to the database. 
This ensures that the units of properties can be easily converted without requiring internal default units.
The default values are summarized in the following dictionary.

.. code-block:: python
 
  >>> distance_unit = "Bohr"

  >>> property_unit_dict = {
  >>>       "energy": "Hartree",
  >>>       "forces": "Hartree/Bohr",
  >>>       "nacs": "1",
  >>>       "dipoles": "1",
  >>>       "socs": "1"
  >>> }      

Moreover, during the generation of a databse, the total number of electronic states as well as the number of singlet, doublet and triplet states are stored in form of metadata to the **SpaiNN** database.

The following python code shows how to generate a **SPaiNN** database (``fancy_name.db``) from SHARC output, *e.g.*, for Wigner sampled geometries (with data from *e.g.* ``ICOND_00000/QM.out`` and ``ICOND_00000/QM.in``) stored in the folder ``data`` using python API:

1. Import Required Packages – To start, you need to import necessary packages. This can be achieved by including the following lines of code:

.. code-block:: python

  >>> import os
  >>> import spainn
  >>> from spainn.asetools import GenerateDB, ConvertDB

2. Generating the Database – To generate the database, you employ the ``generate`` function from the ``GenerateDB`` module of **SPaiNN**. This function extracts data from the ``QM.in`` and ``QM.out`` files. Here's how to perform this step:

.. code-block:: python

  >>> # Specify the path to the folder containing the QM data
  >>> datapath = os.path.join(os.getcwd(), 'data')

  >>> # Instantiate the GenerateDB class
  >>> genDB = GenerateDB()

  >>> # Generate the database
  >>> genDB.generate(path=datapath,
  >>>                dbname=os.path.join(datapath, 'fancy_name.db'),
  >>>                smooth_nacs=True)

  INFO:spainn.asetools.generate_db:Found following state list: ['3']
  INFO:spainn.asetools.generate_db:Found following properties: energy forces dipoles nacs
  INFO:spainn.asetools.generate_db:Wrote 100 geometries to ./data/fancy_name.db

In this step, you provide the path to the folder that holds the Wigner-sampled structures' ``QM.in`` and ``QM.out`` files (either directly or in subfolders). 
The generated database will have a name you define *via* the parameter ``dbname``.
Additionally, you have the option to include both the original non-adiabatic couplings (NACs) and a smoothed version 
(see section :ref:`Nonadiabatic couplings (NACs) <section-nacs>`) within the database by using ``smooth_nacs=True``. 

3. Updating Metadata – Finally, you update the metadata information of the database. This step involves adding details about the reference method, number and multiplicity of electronic states, and property units. Here's how it's done:

.. code-block:: python

  >>> from ase.db import connect

  >>> # Define metadata information (dictionary)
  >>> metadata = {}
  >>> metadata['info'] = '''100 wigner sampled structures of sample molecule'''
  >>> metadata['ReferenceMethod'] = 'SA3-CASSCF(2,2)' # state-average CASSCF with 2 electrons in 2 orbitals
  >>> metadata['_distance_unit'] = 'Bohr'
  >>> metadata['_property_unit_dict'] = {'energy': 'Hartree', 
  >>>                                'forces': 'Hartree/Bohr', 
  >>>                                'nacs': '1', # arb. units
  >>>                                'smooth_nacs': '1', # arb. units
  >>>                                'dipoles': '1'} # arb. units
  >>> metadata['atomrefs'] = {}
  >>> metadata['n_singlets'] = 3 # S0, S1, and S2
  >>> metadata['n_triplets'] = 0 # no triplets
  >>> metadata['phasecorrected'] = False # phase-properties (NACs, dipoles) are not phase corrected
  >>> metadata['states'] = 'S S S' # three singlet states

  >>> # Connect to the database and update metadata
  >>> db = connect(os.path.join(datapath, 'fancy_name.db'))
  >>> db.metadata = metadata

In this step, you define various metadata details such as the reference method used, the units for different properties, the number of singlet and triplet states, and other relevant information. The metadata is then associated with the database.

By following these steps, you'll create a **SPaiNN** database from your quantum chemical calculation results, ready for further analysis and utilization.

The following code snippet shows how to generate the respective database from SHARC output using command line API: 

.. code-block:: console

   (venv)$ spainn-db generate ./data/ ./data/fancy_name.db

This will recursively search "sharc_folder" for `'QM.in'` and `'QM.out'` files, parses them and adds them to `fancy_name.db`. 
Metadata for the database is automatically generated from the QM files.

If you want to train smoothed nonadiabatic couplings :math:`\mathbf{C}_{ij}^s` add -s to calculate and add them to the database right away.

.. code-block:: console

   (venv)$ spainn-db generate ./data/ ./data/fancy_name.db -s

*Note: If you created your database without smooth NACs but want to use them, you can add them afterwards without having to create the database again.*


Convert an existing- into a SPaiNN Database
--------------------------------------------

Conversion of an existing databse, *e.g.*, SchNarc database (*e.g.* `CH2NH2+ <https://github.com/schnarc/SchNarc/blob/master/examples/CH2NH2%2B/CH2NH2%2B.db>`_), 
into a **SpaiNN** database can be performed using the following code snippet.
Noteworthy, in subsequent steps of the data pipeline expect a dictionary of units (properties and atomic coordinates) and states (number of singlet, doublet, and triplet states).
The following code snippet demonstrates how to add this information during the database conversion:

.. rubric:: Python

.. code-block:: python

 >>> import spainn
 >>> from spainn.asetools import ConvertDB
 >>> oldDB = os.path.join(os.getcwd(), 'data', 'schnarc_sample.db')
 >>> newDB = os.path.join(os.getcwd(), 'data', 'spainn_sample.db')
 >>> metadata = {
 >>>  '_distance_unit': 'Bohr', 
 >>>  '_property_unit_dict': {'energy': 'Hartree', 'forces': 'Hartree/Bohr', 'nacs': '1', 'smooth_nacs': '1'},
 >>>  'n_singlets': 3, 'n_doublets': 0, 'n_triplets': 0, 
 >>>  'phasecorrected': False, 'states': 'S S S'
 >>> }
 >>> convDB = ConvertDB()
 >>> convDB.convert(olddb=oldDB, newdb=newDB, copy_metadata=False, smooth_nacs=True)
 >>> db_new = connect(newDB)
 >>> db_new.metadata = metadata

 INFO:spainn.asetools.convert_db:Converting ./data/schnarc_sample.db into ./data/spainn_sample.db
 INFO:spainn.asetools.convert_db: ./data/schnarc_sample.db has 100 entries
 INFO:spainn.asetools.convert_db: ./data/schnarc_sample.db keys: energy has_forces forces dipoles nacs

.. rubric:: Bash

Converting your existing SchNarc database is as simple as generating a new database from SHARC runs. Use spainn-db with the option convert in console.

.. code-block:: console

   (venv)$ spainn-db convert ./data/schnarc_sample.db ./data/spainn_sample.db

This will read data from `schnarc_sample.db` and saves it to `spainn_sample.db` after conversion. 
If you want to train smoothed nonadiabatic couplings :math:`\mathbf{C}_{ij}^s` add -s to calculate and add them to the database right away.

.. code-block:: console

   (venv)$ spainn-db convert ./data/schnarc_sample.db ./data/spainn_sample.db -s

*Note: If you converted your database without smooth NACs but want to use them, you can add them afterwards without having to convert the database again.*

Per default the metadata from `./data/schnarc_sample.db` will be copied to `spainn_sample.db`. 
This can be disabled with the parameter -m:

.. code-block:: console

   (venv)$ spainn-db convert ./data/schnarc_sample.db ./data/spainn_sample.db -m

To add smoothed nonadiabatic couplings to an existing SPaiNN database that containes entries with the keys
`energy` and `nacs`, you can run:

.. code-block:: console

   (venv)$ spainn-db add_smooth spainn.db
