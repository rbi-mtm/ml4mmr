import os
import re
import logging
from typing import Dict, List, Optional, Union, Any

import ase.db
import numpy as np

import spainn
from spainn.properties import SPAINN
from spainn.asetools.aseutils import DatabaseUtils

__all__ = ["GenerateDB"]

log = logging.getLogger(__name__)


class GenerateDB(DatabaseUtils):
    """
    Generates a SPaiNN database from SHARC outputs
    """

    def __init__(self) -> None:
        super().__init__()

        self.property_re = {
            SPAINN.energy: re.compile(r"^!(.*)Hamiltonian"),
            SPAINN.forces: re.compile(r"^!(.*)Gradient"),
            SPAINN.dipoles: re.compile(r"^!(.*)Dipole"),
            SPAINN.nacs: re.compile(r"^!(.*)Non-adiabatic"),
            "skip": re.compile(r"^!(.*)"),
        }

        self.input_re = {
            "states": re.compile(r"^[sS]tates\ (?P<states>.*)"),
            "unit": re.compile(r"^[uU]nit\ (?P<unit>.*)"),
        }
        # State dict and units
        self.n_states = {"n_singlets": 0, "n_duplets": 0, "n_triplets": 0}
        self.unit = "Bohr"

        # Supported properties
        self.property_units = {
            "energy": "Hartree",
            "forces": "Hartree/Bohr",
            "nacs": "1",
            "dipoles": "1",
            "socs": "1",
        }

    def generate(self, path: str, dbname: str, smooth_nacs: bool = False) -> None:
        """
        Searches for all QM.in and QM.out in each subdirectory of given path.
        Make sure only files you want to add to the database are in the path.

        Args:
            path:        path to folders with QM.in and QM.out
            dbname:      name of the resulting database
            smooth_nacs: calculate smooth_nacs
        """
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Directory {path} does not exist.")
        if not dbname:
            raise ValueError("No name for output database given.")
        if os.path.isfile(dbname):
            raise FileExistsError(f"{dbname} already exists.")

        # Check if any QM.in and QM.out files are in the given path
        directories = [x[0] for x in os.walk(path)]
        directories = list(filter(self.check_qm_files, directories))
        assert len(directories) > 0, "No QM.in and QM.out found in given path"

        # Get metadata from QM.in
        metadata = self.parseQMin(os.path.join(directories[0], "QM.in"))
        states = metadata["states"].split()
        for idx, key in enumerate(self.n_states):
            if idx < len(states):
                self.n_states[key] = int(states[idx])

        if metadata["unit"].lower() != "bohr":
            log.critical(
                "Found unit %s, but currently only Bohr is supported.", metadata["unit"]
            )

        # Get properties from QM.out
        props = list(self.parseQMout(os.path.join(directories[0], "QM.out")).keys())

        log.info("Found following state list: %s", states)
        log.info("Found following properties: %s", " ".join(props))

        # Check if smooth NACs requested and if possible to calculate
        if smooth_nacs and not set([SPAINN.energy, SPAINN.nacs]).issubset(set(props)):
            log.critical(
                "smooth_nacs requested but energies or NACs missing. Continue without smooth_nacs"
            )
            smooth_nacs = False

        # Write database
        with ase.db.connect(dbname) as conn:
            conn.metadata = self._build_metadata(props, smooth_nacs)

            for d in directories:
                data = self.parseQMout(os.path.join(d, "QM.out"))
                if smooth_nacs:
                    data["smooth_nacs"] = self.calc_smooth_nacs(
                        data[SPAINN.nacs], data[SPAINN.energy]
                    )
                atom = self.parseQMin(os.path.join(d, "QM.in"))["atom"]
                conn.write(atom, data=data)
        log.info("Wrote %s geometries to %s", len(directories), dbname)

    def add_smooth_nacs(self, dbname: str) -> None:
        """
        Add smooth NACs to existing DB
        """
        if not os.path.isfile(dbname):
            raise FileNotFoundError(f"{dbname} does not exist.")

        with ase.db.connect(dbname) as conn:
            for row in conn.select():
                data = row.data

                if "smooth_nacs" in data:
                    raise KeyError(f"{dbname} already contains smooth NACs")

                data["smooth_nacs"] = self.calc_smooth_nacs(
                    data[SPAINN.nacs], data[SPAINN.energy]
                )
                conn.update(row.id, data=data)

            metadata = conn.metadata
            if "_property_unit_dict" in metadata:
                metadata["_property_unit_dict"]["smooth_nacs"] = "1"
            conn.metadata = metadata

    def parseQMin(self, file: str) -> Dict[str, Union[str, ase.Atoms]]:
        """
        Parse QM.in file, extract geometry, states and units

        Args:
            file: path to QM.in
        """
        qm_in = {}
        # Read file
        file_data = None
        with open(file, "r", encoding="utf-8") as inp:
            file_data = inp.readlines()

        # Extract xyz from file
        natoms = int(file_data[0])
        symbols = []
        coords = []
        for xyz in file_data[2 : natoms + 2]:
            line = xyz.split()
            symbols.append(line[0])
            coords.append(line[1:])
        qm_in["atom"] = ase.Atoms(symbols=symbols, positions=coords)

        # Search for states and position unit in file
        for line in file_data[natoms + 2 :]:
            for key, regex in self.input_re.items():
                line_match = regex.search(line)
                if line_match:
                    if key == "states":
                        qm_in["states"] = line_match.group("states")
                    elif key == "unit":
                        qm_in["unit"] = line_match.group("unit")
        return qm_in

    def parseQMout(
        self, file: str, states_dict: Optional[Dict[str, int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Parse QM.out file and return SPaiNN conform dictionary

        Args:
            file:        Path to QM.out file
            states_dict: Dict of states
        """
        return self._parse_sharc_output(self._read_sharc_output(file), states_dict)

    def _parse_sharc_output(
        self, output: Dict[str, List[str]], states_dict: Optional[Dict[str, int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Parse outputs from QM.out
        TODO: Currently only for singlets! Need to add triplets
        """
        qm_out = {}
        for key, val in output.items():
            if key == SPAINN.energy:
                states = int(val[0][0].split()[0])
                parsed_energies = np.zeros((1, states), dtype=float)

                for i in range(1, states + 1):
                    parsed_energies[0, i - 1] = val[i][0].split()[::2][i - 1]

                if not states_dict:  # assume only singlets
                    energies = parsed_energies
                else:
                    # Remove redundant data, e.g. 1x triplet instead of 3x
                    states = list(states_dict.values())
                    energies = np.zeros((1, sum(states)), dtype=float)
                    energies[0][: states[0]] = parsed_energies[0][: states[0]]
                    energies[0][states[0] : states[0] + states[1]] = parsed_energies[0][
                        states[0] : states[0] + states[1]
                    ]
                    energies[0][states[0] + states[1] : sum(states)] = parsed_energies[
                        0
                    ][states[0] + 2 * states[1] : states[0] + 2 * states[1] + states[2]]

                    if states.count(0) != len(states)-1:
                        #TODO: SOCs
                        pass

                qm_out[SPAINN.energy] = energies

            elif key == SPAINN.forces:
                natoms = int(val[0][0].split()[0])
                states = len(list(filter(lambda x: (len(x) != 1), val)))
                val = [x[0].split() for x in val]
                forces = np.zeros((natoms, states, 3), dtype=float)
                for i in range(states):
                    forces[:, i] = np.array(
                        val[
                            1 * (i + 1)
                            + (i * natoms) : 1 * (i + 1)
                            + ((i + 1) * natoms)
                        ]
                    )
                qm_out[SPAINN.forces] = -forces

            elif key == SPAINN.dipoles and len(val) != 0:
                n_dipoles = int(val[0][0].split()[0])
                val = [x[0].split()[::2] for x in val]
                dipoles = [
                    [val[1 : 1 + n_dipoles]],
                    [val[2 + n_dipoles : 2 + (2 * n_dipoles)]],
                    [val[3 + (2 * n_dipoles) : 3 + (3 * n_dipoles)]],
                ]
                dipoles = np.array(dipoles, dtype=float)
                d_idx = np.triu_indices(n_dipoles, 0)
                qm_out[SPAINN.dipoles] = np.einsum("ijk->jki", dipoles.squeeze())[d_idx]

            elif key == SPAINN.nacs and len(val) != 0:
                natoms = int(val[0][0].split()[0])
                states = int(np.sqrt(len(list(filter(lambda x: (len(x) != 1), val)))))
                elements = len(val) // (1 + natoms)
                val = [x[0].split() for x in val]
                nacs = []
                for i in range(elements):
                    nacs.append(
                        val[
                            1 * (i + 1)
                            + (i * natoms) : 1 * (i + 1)
                            + ((i + 1) * natoms)
                        ]
                    )
                nac_m = np.array(nacs, dtype=float).reshape((states, states, natoms, 3))
                n_idx = np.triu_indices(states, 1)
                qm_out[SPAINN.nacs] = np.einsum("ijk->jik", nac_m[n_idx])

        return qm_out

    def _build_metadata(self, props: List[str], smooth_nacs: bool) -> Dict[str, Any]:
        """
        Build metadata dictionary for new database
        """
        metadata = {}
        metadata["ReferenceMethod"] = "Unknown"
        metadata["_distance_unit"] = self.unit
        metadata["_property_unit_dict"] = {}
        for key, val in self.property_units.items():
            if key in props:
                metadata["_property_unit_dict"][key] = val
        if smooth_nacs:
            metadata["_property_unit_dict"]["smooth_nacs"] = "1"
        metadata["atomrefs"] = {}
        metadata.update(self.n_states)
        return metadata

    def _read_sharc_output(self, file: str) -> Dict[str, List[str]]:
        """
        Read QM.out (SHARC output file)
        """
        file_dict = {
            SPAINN.energy: [],
            SPAINN.forces: [],
            SPAINN.dipoles: [],
            SPAINN.nacs: [],
        }
        file_data = None
        with open(file, "r", encoding="utf-8") as qmfile:
            file_data = qmfile.readlines()

        current_key = "skip"
        skip = True
        for line in file_data:
            key = self._match_property(line)
            if key in file_dict:
                skip = False
                current_key = key
                continue
            if key == "skip":
                skip = True
                continue
            if skip:
                continue

            file_dict[current_key].append(line.replace("\n", "").split("!"))
        return file_dict

    def _match_property(self, line: str) -> str:
        """
        Read single line and return string if key match
        """
        for key, regex in self.property_re.items():
            if regex.search(line):
                return key
        return "good"
