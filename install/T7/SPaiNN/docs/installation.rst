=============
Installation
=============
.. _installation_instruction:

Requirements
--------------

* `Python <http://www.python.org/>`_ (>=3.8)
* `PyTorch <https://pytorch.org/docs/stable/index.html>`_ (>=1.9)
* `PyTorchLightning <https://www.pytorchlightning.ai/>`_ (>=1.9.0)
* `Hydra <https://hydra.cc/>`_ (>=1.1.0)
* `ASE <https://wiki.fysik.dtu.dk/ase/index.html>`_ (>=3.21)

* `SHARC with pysharc <https://sharc-md.org/>`_ (>=3.0)

Installing using pip
----------------------

The most straightforward method to install **SpaiNN** is by using pip, which will automatically fetch the source code from PyPI. To do this, follow these steps:

1. Open your terminal or command prompt.

2. If you prefer, create and activate a virtual environment (recommended) by executing the following commands:

.. code-block:: console

   $ python -m venv .venv
   $ source .venv/bin/activate   # On Windows, use: .venv\Scripts\activate

3. Install **SpaiNN** using pip:

.. code-block:: console

   (.venv) $ pip install spainn

When you install **SpaiNN** using this method, only the core components necessary for model training and usage with `SchNetPack <https://github.com/atomistic-machine-learning/schnetpack>`_ will be included.

If you intend to utilize the interface to `SHARC <https://sharc-md.org/>`_ for molecular dynamics simulations, an additional installation of `SHARC <https://sharc-md.org/>`_, including pysharc, is required. Refer to the `SHARC documentation <https://sharc-md.org/>`_ for detailed instructions on installing SHARC and pysharc. Alternatively, you can follow the following steps (Building from source).

Building from source
---------------------

If you prefer to build **SpaiNN**, `SchNetPack <https://github.com/atomistic-machine-learning/schnetpack>`_, and `SHARC/pysharc <https://sharc-md.org/>`_ from source, follow these steps:

`SHARC 3.0 (including pysharc) <https://sharc-md.org/>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Set up a conda environment with the required dependencies:

.. code-block:: console

   (base) $ conda create -n spainn -c conda-forge python=3.9 numpy scipy h5py six matplotlib python-dateutil pyyaml pyparsing kiwisolver cycler netcdf4 hdf5 h5utils gfortran gcc fftw

2. Activate the conda environment: 

.. code-block:: console

   (base) $ conda activate spainn

3 Clone the `SHARC <https://github.com/sharc-md/sharc>`_ repository:

.. code-block:: console

   (spainn) $ git clone https://github.com/sharc-md/sharc.git

4. Edit the ``Makefile`` in the ``sharc/source`` folder. In **SPaiNN** some computations are performed using **pysharc**. Therefore, you need to set ``USE_PYSHARC`` to ``true`` in the ``source/Makefile``.

.. code-block:: console

   (spainn) $ cd sharc/source
   (spainn) $ vim Makefile

.. code-block:: bash

   USE_PYSHARC := true

Moreover, you have to change line 96 in the file ``source/input_list.f90`` from ``read(nunit,'(A)', iostat=io)`` to ``read(nunit,'(A)', iostat=stat)``.

.. code-block:: console

   (spainn) $ vim input_list.f90

5. Compile and install pysharc:

.. code-block:: console

   (spainn) $ cd ../pysharc/
   (spainn) $ make install

6. Compile and install SHARC:

.. code-block:: console

   (spainn) $ cd ../source/
   (spainn) $ make install

7. Copy the required files to the conda environment:

.. code-block:: console

   (spainn) $ cd ../pysharc/sharc/
   (spainn) $ cp sharc.cpython-39-x86_64-linux-gnu.so ~/anaconda3/envs/spainn/lib/sharc.so
   (spainn) $ cp ../lib/lib*.so ~/anaconda3/envs/spainn/lib/. 

8. Set up the environment:

.. code-block:: console

   (spainn) $ cd ../../bin/
   (spainn) $ source sharcvars.sh

9. Optionally, add the path to SHARC as an environment variable to your ``.bashrc``:

.. code-block:: console

   (spainn) $ vi ~/.bashrc # Add the line: export SHARC=/<path>/sharc/bin/
   (spainn) $ source ~/.bashrc

`SchNetPack <https://github.com/atomistic-machine-learning/schnetpack>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

10. Install `SchNetPack <https://github.com/atomistic-machine-learning/schnetpack>`_

.. code-block:: console

   (spainn) $ pip install --upgrade schnetpack

SPaiNN
^^^^^^^^

11. Install SPaiNN:

.. code-block:: console

   (spainn) $ git clone https://github.com/ExcitedStateML/SPaiNN.git
   (spainn) $ cd SPaiNN
   (spainn) $ pip install .

**Note:** You can replace steps 10 and 11 by simply installing **SpaiNN** through pip, which will automatically get the source code of **SpaiNN** and SchNetPack from PyPI:

.. code-block:: console

   (spainn) $ pip install --upgrade spainn
