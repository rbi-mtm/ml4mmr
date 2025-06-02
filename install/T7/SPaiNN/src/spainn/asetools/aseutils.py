import os
import logging
import numpy as np

__all__ = ["DatabaseUtils"]

log = logging.getLogger(__name__)


class DatabaseUtils:
    """
    Basic functions
    """

    def __init__(self) -> None:
        pass

    def generate_name(self, olddb: str) -> str:
        """
        Add "_new" to file string
        """
        oldname, ext = os.path.splitext(olddb)
        return oldname + "_new" + ext

    def checkpaths(self, olddb: str, newdb: str) -> None:
        """
        Check if valid filepaths are given
        """
        if not os.path.isfile(olddb):
            raise FileNotFoundError(f"{olddb} does not exist.")
        if os.path.isfile(newdb):
            raise FileExistsError(f"{newdb} already exists.")

    def check_qm_files(self, path: str) -> bool:
        """
        Checks if QM.in and QM.out exist in given path
        """
        if os.path.isfile(os.path.join(path, "QM.in")) and os.path.isfile(
            os.path.join(path, "QM.out")
        ):
            return True
        return False

    def calc_smooth_nacs(self, nacs: np.ndarray, energy: np.ndarray) -> np.ndarray:
        """
        Calculate smooth NACs
        smooth_nac = nacs_ij * deltaE_ij
        """
        energy = energy.squeeze()
        idx = np.triu_indices(energy.shape[0], k=1)
        delta_e = energy[idx[0]] - energy[idx[1]]

        return nacs * delta_e[None, :, None]
