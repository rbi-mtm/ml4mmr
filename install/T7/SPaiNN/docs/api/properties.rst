=================
spainn.properties
=================
.. _api_spainn_class:

The SPAINN class
-----------------

.. autoclass:: properties.SPAINN
   :members:


Attributes for Properties
--------------------------

.. rubric:: Properties per electronic state

Fundamental properties that stem directly from a single electronic states (:math:`j`) and are intrinsic to that state without any interaction.

.. autosummary::
   :toctree: generated
   :nosignatures:

   properties.SPAINN.energy
   properties.SPAINN.forces

.. rubric:: Properties per coupling of states

Properties that emerge from the coupling or interaction of two electronic states.

.. autosummary::
   :toctree: generated
   :nosignatures:

   properties.SPAINN.nacs
   properties.SPAINN.smooth_nacs
   properties.SPAINN.dipoles
   properties.SPAINN.socs


Statistics for Properties
--------------------------

The :py:class:`spainn.SPAINN` class is not only used to initialize the default property names,
but further to perform statistical analysis of the training and validation data.
For further details on the statistics module for multiple electronic states refer to `multidatamodule <stats.rst>`_.

.. autosummary::
   :toctree: generated
   :nosignatures:

   properties.SPAINN.get_stats

