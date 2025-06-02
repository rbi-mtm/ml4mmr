=======
Models
=======
.. _usage_models:

Text Text Text

Phase-less Loss
---------------
.. _usage_phaselessloss:

One approach to rectify phase inconsistencies in phase properties (*e.g.*, nonadiabatic couplings) involves monitoring the overlap of wavefunctions across all reference data points. 
An alternative, computationally more efficient approach to surmount the phase predicament is through the utilization of a loss function that disregards phase information. 
The core concept underlying this phase-less loss method is to generate every conceivable phase permutation of the prediction and identify the permutation that exhibits the minimal error in contrast to the intended outcome. 
Considering that the initial phase element can be set to either 1 or -1, the count of potential phase permutations totals :math:`2^{(N_C−1)}` for a system comprising :math:`N_C` couplings.

* **PhaseLossNac:**
  Test Test Test

* **PhaseLossNacMSE:**
  Test Test Test

