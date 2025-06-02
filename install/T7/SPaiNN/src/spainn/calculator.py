import os
from typing import Dict, Union, List

import ase
import numpy as np
import torch
from schnetpack.transform import MatScipyNeighborList

from spainn.interface import NacCalculator
from spainn.properties import SPAINN

__all__ = ["SPaiNNulator"]


class SPaiNNulatorError(Exception):
    """
    SpaiNNulator error class
    """


class ThresholdError(Exception):
    """
    If model threshold exeeded
    """


symbols = ["", "H", "He", "Li", "Be", "B", "C", "N", "O", "F"]


class SPaiNNulator:
    """
    Interface between SHARC and SchNetPack 2.0
    """

    def __init__(
        self,
        atom_types: Union[np.ndarray, torch.Tensor, str] = None,
        modelpath: Union[List[str], str] = "",
        cutoff: float = 10.0,
        properties: List[str] = None,
        n_states: Dict[str, int] = None,
        thresholds: Dict[str, float] = None,
        nac_key: str = SPAINN.nacs,
    ):
        """
        Parameters
        ----------

        atom_types: atomic charges or string of atoms
        modelpath: path(s) to trained model(s) or folder(s) with 'best_inference_model'
            for adaptive sampling
        cutoff: cutoff value
        properties: list of properties returned to SHARC
        n_states: dictionary of calculated states
        thresholds: dictionary of threshold values
        
        Examples
        ---------

        You can use this calculator to perform predictions of properties using a 
        trained NN model.

        Here we show, how to predict the energies for a target molecule. First we
        import all necessary modules followed by the creation of an `ase.Atoms` 
        object of the target molecule.

        >>> import os, sys
        >>> import numpy as np
        >>> import ase
        >>> 
        >>> symbols = 'CNHHHH'
        >>> positions = np.array(
        >>>     [[ 0.0000,  0.0000,  0.0000 ],
        >>>      [ 2.4321,  0.0000,  0.0000 ],
        >>>      [-1.0111,  1.7951,  0.0000 ],
        >>>      [ 3.4373,  1.6202, -0.2566 ],
        >>>      [ 3.4373, -1.6202,  0.2566 ],
        >>>      [-1.0111, -1.7951,  0.0000 ]]
        >>> )
        >>> # create ase Atoms object
        >>> target_mol = ase.Atoms(symbols=symbols, positions=positions)

        Next, we define the calculator `NacCalculator` used to predict the energy
        of the target molecule and perform the prediction.

        >>> from schnetpack.transform import MatScipyNeighborList
        >>> from spainn.interface import NacCalculator
        >>> 
        >>> calc = NacCalculator(
        >>>     model_file=os.path.join(os.getcwd(), 'train', 'best_model'),
        >>>     neighbor_list=MatScipyNeighborList(cutoff=10.0)
        >>> )
        >>> target_mol.calc = calc
        >>> # make prediction
        >>> pred = target_mol.get_properties(['energy'])
        """

        if atom_types is None:
            raise SPaiNNulatorError("atom_types has to be set")

        # Load model and setup molecule
        self.modelpath = [modelpath] if isinstance(modelpath, str) else modelpath
        self.properties = (
            properties
            if properties is not None
            else [
                SPAINN.energy,
                SPAINN.forces,
                SPAINN.nacs,
                SPAINN.dipoles,
            ]
        )

        if isinstance(atom_types, str):
            atom_types = np.array([symbols.index(c) for c in atom_types.upper()])

        self.molecule = [
            ase.Atoms(symbols=atom_types) for _ in range(len(self.modelpath))
        ]
        self.thresholds = thresholds
        self.atom_types = atom_types
        self.nac_key = nac_key

        self._check_modelpath()

        # Setup states and matrix masks
        if n_states is None:
            raise SPaiNNulatorError("n_states dict has to be set!")
        self.n_states = n_states
        self.n_total_states = n_states["n_singlets"] + 3 * n_states["n_triplets"]
        self.n_atoms = len(atom_types)

        self.nac_idx = np.triu_indices(self.n_states["n_singlets"], 1)
        self.dm_idx = np.triu_indices(self.n_states["n_singlets"], 0)
        self.soc_idx = np.triu_indices(self.n_total_states, 1)

        self.last_prediction = None

        # Use NacCalculator to calculate properties
        for idx, val in enumerate(self.modelpath):
            self.molecule[idx].calc = NacCalculator(
                model_file=val,
                neighbor_list=MatScipyNeighborList(cutoff=cutoff),
                energy=SPAINN.energy,
                forces=SPAINN.forces,
            )

    def calculate(
        self, sharc_coords: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, List[np.ndarray]]:
        """
        Calculate properties from new positions.

        If multiple models are used, the average values between
        the two predictions with the lowest NAC MAE will be returned

        Parameters
        ----------

        sharc_coords: Coordinates from SHARC simulation
        """

        spainn_output = []
        for i in range(len(self.modelpath)):
            self.molecule[i].set_positions(sharc_coords)
            spainn_output.append(self.molecule[i].get_properties(self.properties))

        # Save first prediction for phase tracking
        if self.last_prediction is None:
            self.last_prediction = spainn_output[0]

        if len(self.modelpath) == 1:
            for prop in self.properties:
                if prop not in ["energy", "forces"]:
                    spainn_output[0][prop] = self._adjust_phase(
                        self.last_prediction[prop], spainn_output[0][prop]
                    )
            self.last_prediction = spainn_output[0]
            return self.get_qm(spainn_output[0])

        # Adjust phases relative to first model
        for prop in self.properties:
            if prop not in ["energy", "forces"]:
                for idx, val in enumerate(spainn_output):
                    spainn_output[idx][prop] = self._adjust_phase(
                        self.last_prediction[prop], spainn_output[idx][prop]
                    )

        # Check if Thresholds exeeded
        if self.thresholds is not None:
            prop_mae = {
                key: np.mean(np.abs(val - spainn_output[1][key]))
                for (key, val) in spainn_output[0].items()
            }
            below_threshold = all(prop_mae[k] < v for (k, v) in self.thresholds.items())
            if not below_threshold:
                self._write_xyz(sharc_coords)
                raise ThresholdError("Threshold exeeded.")

        # Save last prediction
        self.last_prediction = spainn_output[0]
        return self.get_qm(spainn_output[0])

    def get_qm(self, spainn_output: List[np.ndarray]) -> Dict[str, List[np.ndarray]]:
        """
        Calculate QM string for SHARC with predictions from model.
        """

        states = self.n_total_states
        n_singlets = self.n_states["n_singlets"]
        n_triplets = self.n_states["n_triplets"]

        qm_out = {}
        # Convert energy array to complex diagonal matrix
        qm_out["h"] = np.diag(np.array(spainn_output["energy"], dtype=complex)).tolist()

        # Reshape force array from [atoms, states, coords] to [states, atoms, coords]
        qm_out["grad"] = np.einsum("ijk->jik", -spainn_output["forces"]).tolist()

        if self.nac_key in self.properties:
            nacs_v = np.einsum("ijk->jik", spainn_output[self.nac_key])
            nacs_m = np.zeros((states, states, self.n_atoms, 3))

            if n_triplets == 0:
                nacs_m[self.nac_idx] = nacs_v
                nacs_m -= np.transpose(nacs_m, axes=(1, 0, 2, 3))
            else:
                nacs_singlet = np.zeros((n_singlets, n_singlets, self.n_atoms, 3))
                nacs_singlet[self.nac_idx] = nacs_v[
                    0 : int(n_singlets * (n_singlets - 1) / 2)
                ]
                nacs_singlet -= nacs_singlet.T

                nacs_m[0:n_singlets, 0:n_singlets] = nacs_singlet

                nacs_trip_sub = np.zeros((n_triplets, n_triplets, self.n_atoms, 3))
                sub_idx = np.triu_indices(n_triplets, 1)
                nacs_trip_sub[sub_idx] = nacs_v[int(n_singlets * (n_singlets - 1) / 2) :]
                nacs_trip_sub -= nacs_trip_sub.T

                nacs_trip = np.zeros((3 * n_triplets, 3 * n_triplets, self.n_atoms, 3))

                for i in range(3):
                    for j in range(i, 3):
                        nacs_trip[
                            i * n_triplets : (i + 1) * n_triplets,
                            j * n_triplets : (j + 1) * n_triplets,
                        ] = nacs_trip_sub

                trip_idx = np.tril_indices(3 * n_triplets)
                nacs_trip[trip_idx] = 0
                nacs_trip -= nacs_trip.T

                nacs_m[n_singlets:, n_singlets:] = nacs_trip

            qm_out["nacdr"] = nacs_m.tolist()

        if "dipoles" in self.properties:
            dm_m = np.zeros((states, states, 3), dtype=complex)
            if n_triplets == 0:
                dm_m[self.dm_idx] = spainn_output["dipoles"]
                dm_m += dm_m.T
                dm_m[self.dm_idx] = spainn_output["dipoles"]
            else:
                dm_singlets = np.zeros((n_singlets, n_singlets, 3), dtype=complex)
                dm_singlets[self.dm_idx] = spainn_output["dipoles"][
                    : int(n_singlets * (n_singlets - 1) / 2) + n_singlets
                ]
                dm_singlets += dm_singlets.T
                dm_singlets[self.dm_idx] = spainn_output["dipoles"][
                    : int(n_singlets * (n_singlets - 1) / 2) + n_singlets
                ]

                dm_m[0:n_singlets, 0:n_singlets] = dm_singlets

                dm_triplets = np.zeros((n_triplets, n_triplets, 3), dtype=complex)
                trip_idx = np.triu_indices(n_triplets, 0)
                dm_triplets[trip_idx] = spainn_output["dipoles"][
                    int(n_singlets * (n_singlets - 1) / 2) + n_singlets :
                ]
                dm_triplets += dm_triplets.T
                dm_triplets[trip_idx] = spainn_output["dipoles"][
                    int(n_singlets * (n_singlets - 1) / 2) + n_singlets :
                ]

                for i in range(0, 3):
                    dm_m[
                        n_singlets + i * n_triplets : n_singlets + i * n_triplets
                    ] = dm_triplets

            dm_m = np.einsum("ijk->kij", dm_m)
            qm_out["dm"] = dm_m.tolist()

        if "socs" in self.properties:
            soc_m = np.zeros((states, states), dtype=complex)
            soc_m[self.soc_idx] = spainn_output["socs"]
            soc_m += soc_m.T

            qm_out["h"] += soc_m

        return qm_out

    def _check_modelpath(self) -> None:
        """
        Check if valid path(s) given.
        """
        for path in self.modelpath:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"'{path}' does not exist!")

    def _adjust_phase(
        self, primary_phase: np.ndarray, secondary_phase: np.ndarray
    ) -> np.ndarray:
        """
        Function to align the phases of the two predictions.
        """

        # Make sure only NACS are transformed
        is_nac = bool(len(primary_phase.shape) > 2)
        if is_nac:
            primary_phase = np.einsum("ijk->jik", primary_phase)
            secondary_phase = np.einsum("ijk->jik", secondary_phase)

        # Adjust phases
        for idx, val in enumerate(secondary_phase):
            if np.vdot(val, primary_phase[idx]) < 0:
                secondary_phase[idx] *= -1
        return np.einsum("ijk->jik", secondary_phase) if is_nac else secondary_phase

    def _write_xyz(self, coords: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Write rejected geometry to xyz file.
        """
        with open("aborted.xyz", "w", encoding="utf-8") as output:
            output.write(f"{self.n_atoms}\n")
            output.write("Rejected geometry\n")
            for idx, val in enumerate(coords):
                output.write(
                    f"{symbols[self.atom_types[idx]]}\t{val[0]:12.8f}\t{val[1]:12.8f}\t{val[2]:12.8f}\n"
                )
