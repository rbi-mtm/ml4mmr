from typing import Dict, List, Union

from sharc.pysharc.interface import SHARC_INTERFACE

import spainn
from spainn.calculator import SPaiNNulator


class SHARC_NN(SHARC_INTERFACE):
    """
    Class for SHARC NN
    """
    # Name of the interface
    interface = 'NN'
    # store atom ids
    save_atids = True
    # store atom names
    save_atnames = True
    # accepted units:  0 : Bohr, 1 : Angstrom
    iunit = 0
    # not supported keys
    not_supported = ['nacdt', 'dmdr']

    def __init__(
        self,
        modelpath: str = 'best_inference_model',
        atoms: Union[List[int], str] = None,
        **kwargs
        ):
        self.spainn_init = SPaiNNulator(atom_types=atoms, modelpath=modelpath,**kwargs)

    def initial_setup(self, **kwargs):
        pass

    def do_qm_job(self, tasks, Crd):
        return self.spainn_init.calculate(Crd)


    def final_print(self):
        self.sharc_writeQMin()

    def readParameter(self, param,  *args, **kwargs):
        pass
