from typing import Tuple, Final
from schnetpack.data import AtomsDataModule
import torch

from spainn.multidatamodule import calculate_multistats

__all__ = ["SPAINN"]

class SPAINN(AtomsDataModule):
    """
    Adapted AtomsDataModule class of SchNetPack 2.0 for calculating 
    statistics (mean and standard deviation) for multiple electronic
    states.

    The total number of electronic states (`n_states`) refers to the 
    total number including every multiplicity. It can be calculated in
    the following way:

    .. math::

        N_{\\text{states}} = 1\\cdot N_{\\text{singlets}} 
            + 2\\cdot N_{\\text{doublets}} 
            + 3\\cdot N_{\\text{triplets}} + \\ldots

    The total number of couplings (`n_nacs`) can be computed from the 
    number of electronic states according to:

    .. math::

        N_{\\text{couplings}} = \\frac{1}{2}N_{\\text{singlets}}\\left(N_{\\text{singlets}}-1\\right)
            + \\frac{1}{2}N_{\\text{doublets}}\\left(N_{\\text{doublets}}-1\\right)
            + \\ldots

    """

    # Keys for properties used throughout whole workflow
    energy: Final[str] = "energy" #: label of energies
    forces: Final[str] = "forces" #: label for forces
    nacs: Final[str] = "nacs" #: label for nonadiabatic couplings
    dipoles: Final[str] = "dipoles" #: label for dipoles
    socs: Final[str] = "socs" #: label for spin-orbit couplings
    smooth_nacs: Final[str] = "smooth_nacs" #: label for smoothed nonadiabatic couplings

    def __init__(
            self, 
            n_nacs: int, 
            n_states: int, 
            **kwargs
            ):
        """
        Args:        
            n_states: :math:`N_{\\text{states}}` - number of electronic states
            n_nacs: :math:`N_{\\text{couplings}}` - number of couplings
            datapath: path to dataset
            batch_size: batch size for training
            num_train: number of training examples (absolute or relative). If None, 
                       the number is obtained from num_val and num_test
            num_val: number of validation examples (absolute or relative). If None,
                     the number is obtained from num_train and num_test
            num_test: number of test examples (absolute or relative). If None,
                      the number obtained from num_train and num_val.
            split_file: path to npz file with data partitions
            format: format of the dataset (*e.g.* ASE)
            load_properties: subset of properties to load
            val_batch_size: validation batch size. If None, use test_batch_size, then
                            batch_size.
            test_batch_size: test batch size. If None, use val_batch_size, then
                             batch_size.
            transforms: Preprocessing transform applied to each system separately before
                        batching.
            train_transforms: Overrides transform_fn for training.
            val_transforms: Overrides transform_fn for validation.
            test_transforms: Overrides transform_fn for testing.
            num_workers: Number of data loader workers.
            num_val_workers: Number of validation data loader workers.
            num_test_workers: Number of test data loader workers.
            property_units: Dictionary from property to corresponding unit as a string, *e.g.*,
                            eV or kcal/mol.
            distance_unit: Unit of the atom positions and cell as a string, *e.g.*,
                           Ang or Bohr.
            data_workdir: Copy data here as part of setup, *e.g.*, to a local file
                          system for faster performance.
            cleanup_workdir_stage: Determines after which stage to remove the data
                                   workdir
            splitting: Method to generate train/validation/test partitions.
            pin_memory: If true, pin memory of loaded data to GPU. Default: Will be
                        set to true, when GPUs are used.


        Examples
        --------

        >>> import sys, os
        >>> import schnetpack as spk
        >>> import spainn

        Create a AtomsDataModule with :py:class:`SPAINN` (minimum example):

        >>> data_module = spainn.SPAINN(
        >>>     n_states=2, # 2 electronic states
        >>>     n_nacs=1, # one coupling between state 1 and 2
        >>>     datapath=os.path.join(os.getcwd(), 'database.db'),
        >>>     batch_size=2,
        >>>     num_train=0.6,
        >>>     num_val=0.1,
        >>> )

        Prepare and setup the data

        >>> data_module.prepare_data()
        >>> data_module.setup()

        Note: The `get_stats` function within :py:class:`spainn.SPAINN` replaces the `get_stats` function
        inherited from the SchNetPack module :py:class:`AtomsDataModule`. However, it works also for 
        single electronic states.
        """

        self.n_nacs = n_nacs
        self.n_states = n_states
        super().__init__(**kwargs)

    def get_stats(self, 
            property: str, 
            divide_by_atoms: bool, 
            remove_atomref: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Statistics of multiple electronic states.
        
        Calculate or access already computed statistics, *i.e.*, mean and standard
        deviation of a selected property for multiple electronic states.
        Note: The statistics is seperately computed for every electronic state.

        Parameters
        ----------

        divide_by_atoms: dict from property name to bool. If True, 
            divide property by number of atoms before calculating statistics.
        remove_atomref: If true, remove reference values for single atoms 
            before calculating stats.
        key: Key of property, for which statistics is computed or returned (if
            already computed and stored in `_stats` dictionary).

        Returns
        -------

        Dictionary of computed statistics, *i.e*, mean and standard deviation of a 
            selected property.
        """

        # define key for computing statistics
        key = (property, divide_by_atoms, remove_atomref)
        
        # if statistics already calculated return stored values
        if key in self._stats:
            return self._stats[key]

        # compute mean and standard deviation of multiple states
        stats = calculate_multistats(
            self.train_dataloader(),
            divide_by_atoms={property: divide_by_atoms},
            atomref=self.train_dataset.atomrefs,
            n_states=self.n_states,
            n_nacs=self.n_nacs,
        )[property]
        self._stats[key] = stats

        return stats

