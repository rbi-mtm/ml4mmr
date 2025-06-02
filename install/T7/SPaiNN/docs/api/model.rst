=============
spainn.model
=============

Properties per electronic state
--------------------------------

Prediction modules for fundamental properties that stem directly from a single electronic states (:math:`j`) and 
are intrinsic to that state without any interaction.

Energies
^^^^^^^^^ 

.. autoclass:: model.Atomwise
   :members:

Forces
^^^^^^^

.. autoclass:: model.Forces
   :members:

Properties per coupling of states
----------------------------------

Prediction modules for properties that emerge from the coupling or interaction of two electronic states.

Nonadiabatic couplings
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: model.Nacs
   :members:

Dipoles Moments
^^^^^^^^^^^^^^^^

.. autoclass:: model.Dipoles
   :members:

Spin-orbit couplings
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: model.Socs
   :members:


