from typing import List, Union

import numpy as np
import schnetpack
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from schnetpack.units import convert_units

import spainn
from spainn.properties import SPAINN

__all__ = ["NacCalculator"]


class AtomsConverterError(Exception):
    pass


class NacCalculator(Calculator):
    """
    Adapted SpkCalculator for predicting NACs (and other properties)

    Args:
        model_file:     Path to trained model
        neighbor_list:  Neighborlist transform
        energy_key:     Name of energy property in provided model
        force_key:      Name of force property in provided model
        nac_key:        Name of NAC property in provided model
        dipole_key:     Name of dipole property in provided model
        soc_key:        Name of SOC property in provided model
        smooth_nac_key: Name of smooth NAC property in provided model
        energy_unit:    Energy unit used by model
        position_unit:  Unit used for positions
        soc_unit:       SOC unit used by model
        dipole_unit:    Dipole unit used by model
        nac_unit:       NAC unit used by model, default eV to avoid
                        conversion by schnetpack
        device:         Device on which model operates
        dtype:          Data type for model input
        converter:      Converter used to set up input batches
        **kwargs: Additional arguments for Calculator class
    """

    energy = "energy"
    forces = "forces"
    nacs = "nacs"
    dipoles = "dipoles"
    socs = "socs"
    smooth_nacs = "smooth_nacs"
    implemented_properties = [energy, forces, nacs, dipoles, socs, smooth_nacs]

    def __init__(
        self,
        model_file: str,
        neighbor_list: schnetpack.transform.Transform,
        energy_key: str = SPAINN.energy,
        force_key: str = SPAINN.forces,
        nac_key: str = SPAINN.nacs,
        dipole_key: str = SPAINN.dipoles,
        soc_key: str = SPAINN.socs,
        smooth_nac_key: str = SPAINN.smooth_nacs,
        energy_unit: Union[str, float] = "Ha",
        position_unit: Union[str, float] = "Bohr",
        soc_unit: str = "eV",
        dipole_unit: str = "eV",
        nac_unit: str = "eV",
        snac_unit: str = "eV",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float64,
        converter: schnetpack.interfaces.AtomsConverter = schnetpack.interfaces.AtomsConverter,
        **kwargs,
    ):
        #  Add NACS to SpkCalculator
        Calculator.__init__(self, **kwargs)

        self.converter = converter(neighbor_list, device=device, dtype=dtype)

        self.energy_key = energy_key
        self.force_key = force_key
        self.nac_key = nac_key
        self.dipole_key = dipole_key
        self.soc_key = soc_key
        self.smooth_nac_key = smooth_nac_key

        self.property_map = {
            self.energy: self.energy_key,
            self.forces: self.force_key,
            self.nacs: self.nac_key,
            self.dipoles: self.dipole_key,
            self.socs: self.soc_key,
            self.smooth_nacs: self.smooth_nac_key,
        }

        self.model = self._load_model(model_file)
        self.model.to(device=device, dtype=dtype)

        self.energy_conversion = convert_units(energy_unit, "Ha")
        self.position_conversion = convert_units(position_unit, "Bohr")

        self.property_units = {
            self.energy: self.energy_conversion,
            self.forces: self.energy_conversion / self.position_conversion,
            self.nacs: convert_units(nac_unit, "eV"),
            self.dipoles: convert_units(dipole_unit, "eV"),
            self.socs: convert_units(soc_unit, "eV"),
            self.smooth_nacs: convert_units(snac_unit, "eV"),
        }

        self.model_results = None

    def _load_model(self, model_file: str) -> schnetpack.model.AtomisticModel:
        """
        Load an individual model, activate stress computation
        Args:
            model_file (str): path to model.
        Returns:
           AtomisticTask: loaded schnetpack model
        """

        # load model and keep it on CPU, device can be changed afterwards
        model = torch.load(model_file, map_location="cpu")
        model = model.eval()

        return model

    def calculate(
        self,
        atoms: Atoms = None,
        properties: List[str] = None,
        system_changes: List[str] = all_changes,
    ):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): select properties computed and stored to results.
            system_changes (list of str): List of changes for ASE.
        """
        properties = properties if isinstance(properties, List) else ["energy"]
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)

        if self.calculation_required(atoms, properties):
            Calculator.calculate(self, atoms)

            # Convert to schnetpack input format
            model_inputs = self.converter(atoms)
            model_results = self.model(model_inputs)

            results = {}
            # TODO: use index information to slice everything properly
            for prop in properties:
                model_prop = self.property_map[prop]

                if model_prop in model_results:
                    if model_prop == self.energy:
                        # ase calculator should return scalar energy
                        results[prop] = (
                            model_results[model_prop].detach().numpy()[0]
                            * self.property_units[prop]
                        )
                    elif model_prop == self.smooth_nacs:
                        n_nacs = model_results[self.smooth_nac_key].shape[1]
                        n_nacs = n_nacs if n_nacs != 1 else 2
                        idx = torch.triu_indices(n_nacs, n_nacs, offset=1)
                        energy = model_results[self.energy_key][0]
                        de = energy[idx[0]] - energy[idx[1]]

                        results[prop] = (
                            (model_results[self.smooth_nac_key] / de[None, :, None])
                            .detach()
                            .numpy()
                        )
                    else:
                        results[prop] = (
                            model_results[model_prop].detach().numpy()
                            * self.property_units[prop]
                        )
                else:
                    raise AtomsConverterError(
                        "'{:s}' is not a property of your model. Please "
                        "check the model "
                        "properties!".format(prop)
                    )

            self.results = results
            self.model_results = model_results

    def calculate_properties(self, atoms: Atoms, properties: List[str]) -> np.ndarray:
        """
        Wrapper function to return NACs (and other properties)
        Called by ase.Atoms.get_properties()
        This is the easiest approach to add NAC prediction without touching ase.Atoms class
        """
        self.calculate(atoms, properties)
        return self.results
