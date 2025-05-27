from typing import List, Dict
import numpy as np


HARTREE_TO_KCAL = 627.50961
BOHR_TO_ANGSTROM = 0.529177249

class SharcTrajectoryParser:
    """
    Parser for SHARC trajectory output files. Parses the header on initialization.

    Attributes:
        filepath (str): Path to the SHARC trajectory output file.
        header_data (Dict[str, any]): Dictionary containing parsed header information.
        atomic_numbers (List[int]): List of atomic numbers.
        elements (List[str]): List of element symbols.
        atomic_masses (List[float]): List of atomic masses.
        lines (List[str]): lines from the trajectory file
        
    """
    _mult_to_letter = {0: "S",
                       1: "D",
                       2: "T",
                       3: "Q",
                       4: "P"}

    def __init__(self, filepath: str, to_kcal: bool = True) -> None:
        """
        Initialize the parser and immediately parse the header.

        Args:
            filepath (str): Path to the SHARC output file.
        """
        self.filepath: str = filepath
        self.header_data: Dict[str, any] = {}
        self.atomic_numbers: List[int] = []
        self.elements: List[str] = []
        self.atomic_masses: List[float] = []
        self.lines: List[str] = []
        self._parse_header()
        self.size_hamiltonian: int = sum([(idx+1)*i for idx,i in enumerate(self.header_data['nstates_m'])])
        # self.state_labels: List[str] = [] 
        # for idx,i in enumerate(self.header_data['nstates_m']):
        #     for j in range(i):
        #         if idx < 2:
        #             self.state_labels.append(f"{self._mult_to_letter[idx]}{j}")
        #         else:
        #             self.state_labels.append(f"{self._mult_to_letter[idx]}{j+1}")
                    
        self.conversion = {"energy": 1.0, 
                           "grad": 1.0,
                           "NAC": 1.0,
                           "geom": 1.0}
        if to_kcal:
            self.conversion["energy"] = HARTREE_TO_KCAL
            self.conversion["grad"] = HARTREE_TO_KCAL / BOHR_TO_ANGSTROM
            self.conversion["NAC"] = 1 / BOHR_TO_ANGSTROM
            self.conversion["geom"] = BOHR_TO_ANGSTROM
        

    def _parse_header(self) -> None:
        """
        Parses the header section of the SHARC trajectory file and extracts:
        - Basic simulation parameters (e.g. nstates, dtstep, etc.)
        - Atomic numbers, elements, and masses
        Stores results in instance attributes.
        """
        with open(self.filepath, 'r') as f:
            self.lines = f.readlines()

        i: int = 0
        while i < len(self.lines):
            line: str = self.lines[i].strip()

            if line.startswith("SHARC_version"):
                self.header_data["SHARC_version"] = line.split()[-1]

            elif line.startswith("maxmult"):
                self.header_data["maxmult"] = int(line.split()[-1])

            elif line.startswith("nstates_m"):
                self.header_data["nstates_m"] = list(map(int, line.split()[1:]))

            elif line.startswith("natom"):
                self.header_data["natom"] = int(line.split()[-1])

            elif line.startswith("ezero"):
                self.header_data["ezero"] = float(line.split()[-1])

            elif line.startswith("! Atomic numbers"):
                self.atomic_numbers = [
                    int(float(self.lines[i + j + 1].strip()))
                    for j in range(self.header_data["natom"])
                ]
                i += self.header_data["natom"]

            elif line.startswith("! Elements"):
                self.elements = [
                    self.lines[i + j + 1].strip()
                    for j in range(self.header_data["natom"])
                ]
                i += self.header_data["natom"]

            elif line.startswith("! Atomic masses"):
                self.atomic_masses = [
                    float(self.lines[i + j + 1].strip())
                    for j in range(self.header_data["natom"])
                ]
                i += self.header_data["natom"]

            elif line.startswith("********************************* End of header array data"):
                break

            i += 1

        self.header_data["atomic_numbers"] = np.array(self.atomic_numbers).reshape(-1, 1)
        self.header_data["elements"] = self.elements
        self.header_data["atomic_masses"] = self.atomic_masses
        
    def _split_into_steps(self) -> list[list[str]]:
        """
        Splits the trajectory lines into separate blocks, one for each time step.

        Returns:
            A list of blocks, each corresponding to one time step.
        """
        step_indices = [i for i, line in enumerate(self.lines) if line.strip().startswith("! 0 Step")]
        step_blocks = []

        for i, start in enumerate(step_indices):
            end = step_indices[i + 1] if i + 1 < len(step_indices) else len(self.lines)
            step_blocks.append(self.lines[start:end])

        return step_blocks
    

    def _parse_step(self, block: list[str]) -> dict:
        """
        Parses a single time step block and extracts relevant information.

        Args:
            block: Lines corresponding to one trajectory step.

        Returns:
            A dictionary containing parsed data.
        """
        data = {}
        
        def parse_matrix(start_idx: int, nrows: int, complex_matrix: bool = False) -> list:
            """
            Extracts the data from every block as matrix

            Returns:
                A matrix of the respective property
            """
            matrix = []
            for i in range(start_idx, start_idx + nrows):
                parts = block[i].strip().split()
                if complex_matrix:
                    row = [float(parts[j]) + 1j * float(parts[j + 1]) for j in range(0, len(parts), 2)]
                else:
                    row = [float(x) for x in parts]
                matrix.append(row)
            return np.array(matrix)

        i = 0
        while i < len(block):
            line = block[i].strip()

            if line.startswith("! 1 Hamiltonian (MCH) in a.u."):
                hamiltonian_mat = parse_matrix(start_idx=i + 1, 
                                               nrows=self.size_hamiltonian, 
                                               complex_matrix=True)
                # for energy, label in zip(np.diag(hamiltonian_mat).real, self.state_labels):
                #     data[f"energy_{label}"] = energy * self.conversion['energy']
                for idx, energy in enumerate(np.diag(hamiltonian_mat).real):
                    data[f"energy_{idx}"] = energy * self.conversion['energy']
                i += 1 + self.size_hamiltonian
                

#             if line.startswith("DIPOLE MOMENTS"):
#                 ndir = 3  # x, y, z
#                 data["dipole_moments"] = parse_matrix(i + 1, ndir, complex_matrix=True)
#                 i += 1 + ndir
#                 continue

            elif line.startswith("! 11 Geometry in a.u."):
                geometry = (parse_matrix(start_idx=i + 1, nrows=self.header_data['natom'])
                            * self.conversion['geom'])
                data["nxyz"] = np.hstack([self.header_data['atomic_numbers'], geometry])
                i += 1 + self.header_data['natom']
                

            elif line.startswith("! 15 Gradients (MCH) State"):
                state_no = int(line.split()[-1].strip()) - 1
                grad = (parse_matrix(start_idx=i + 1, 
                                     nrows=self.header_data['natom'])
                        * self.conversion['grad'])
                # data[f"energy_{self.state_labels[state_no]}_grad"] = grad
                data[f"energy_{state_no}_grad"] = grad
                i += 1 + self.header_data['natom']
                

            elif line.startswith("! 16 NACdr matrix element (MCH)"):
                split = line.split()
                bra_state = int(split[-2].strip()) -1 
                ket_state = int(split[-1].strip()) -1
                if bra_state == ket_state:
                    i += 1 + self.header_data['natom']
                    continue
                NACdr = (parse_matrix(start_idx=i + 1, 
                                      nrows=self.header_data['natom']) 
                         * self.conversion['NAC'])
                #data[f"NAC_{self.state_labels[bra_state]}_to_{self.state_labels[ket_state]}"] = NACdr
                data[f"NACdr_{bra_state}{ket_state}"] = NACdr
                data[f"force_nacv_{bra_state}{ket_state}"] = - NACdr * (data[f"energy_{bra_state}"] - data[f"energy_{ket_state}"])
                i += 1 + self.header_data['natom']
                
            else:
                i += 1

        return data
    
    def parse(self) -> List[Dict[str, any]]:
        """
        Parses the full SHARC trajectory file into structured data for each time step.

        This method splits the trajectory file into individual steps and extracts relevant
        information from each step, including the Hamiltonian matrix, dipole moments,
        molecular geometry, energy gradients, and nonadiabatic coupling vectors.

        Returns:
            A list of dictionaries, each corresponding to one time step. 
        """
        step_blocks = self._split_into_steps()
        results : List[Dict[str, any]] = []
        for block in step_blocks:
            results.append(self._parse_step(block))

        return results


