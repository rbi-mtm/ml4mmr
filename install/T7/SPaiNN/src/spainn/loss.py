from typing import Sequence, Dict, Union, Optional, Callable, List, Tuple
import numpy as np
import itertools
from itertools import permutations

import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F

__all__ = [
    "MeanVectorLoss",
    "PhaseException",
    "PhaseLossAtomistic",
    "PhaseLossAtomisticMAE",
    "PhaseLossAtomisticMSE",
    "PhaseLoss",
    "PhaseLossMAE",
    "PhaseLossMSE",
    "PhysPhaseLossAtomistic",
    "PhysPhaseLoss"
]

class MeanVectorLoss(nn.Module):
    """
    A custom loss function for vectorial properties that penalize deviations
    between the direction of the real vector *vs.* the target vector.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        # Calculate MSE
        mse = F.mse_loss(inputs, targets)

        # Cosine similarity to penalize wrong directions
        # 0 = perfect match, 1 = different, 2 = opposite
        cosi = torch.mean(1.0 - F.cosine_similarity(inputs, targets, dim=1))

        # Multiply with 1e-7 in case mse is 0 (uncomment line below)
        # return mse.add(cosi.mul(max(mse,1e-7)))
        return mse.add(cosi.mul(mse))

    def clone(self) -> "Loss":
        # Make a copy of the loss.
        return deepcopy(self)

class PhaseException(Exception):
    pass

class PhaseLossAtomistic(nn.Module):
    """
    A custom loss function for phase loss.
    The loss of each element multiplied by 1 or -1 is taken,
    the lowest value gets returned.
    Basicly a lossless MAE and MSE implementation
    """

    def __init__(self, atoms: int = None, mse=True):
        super().__init__()
        self.atoms = atoms
        self.mse = mse

    def forward(self, inputs, targets) -> torch.Tensor:
        # calculate MAE
        target_states = torch.reshape(
            targets,
            (
                -1,
                self.atoms,
                targets.shape[1],
                targets.shape[2],
            ),
        )
        input_states = torch.reshape(
            inputs,
            (
                -1,
                self.atoms,
                inputs.shape[1],
                inputs.shape[2],
            ),
        )

        if self.mse:
            positive = torch.sum(torch.square(target_states + input_states), dim=(1, 3))
            negative = torch.sum(torch.square(target_states - input_states), dim=(1, 3))
        else:
            positive = torch.sum(torch.abs(target_states + input_states), dim=(1, 3))
            negative = torch.sum(torch.abs(target_states - input_states), dim=(1, 3))

        result = torch.min(positive, negative)
        return torch.sum(result) / targets.numel()

    def clone(self) -> "Loss":
        # Make a copy of the loss.
        return deepcopy(self)


class PhaseLossAtomisticMAE(nn.Module):
    """
    A custom loss function for phase loss.
    The loss of each element multiplied by 1 or -1 is taken,
    the lowest value gets returned.
    Basicly a lossless MAE implementation
    """

    def __init__(self, atoms: int = None):
        super().__init__()
        self.atoms = atoms

    def forward(self, inputs, targets) -> torch.Tensor:
        # calculate MAE
        target_states = torch.reshape(
            targets,
            (
                -1,
                self.atoms,
                targets.shape[1],
                targets.shape[2],
            ),
        )
        input_states = torch.reshape(
            inputs,
            (
                -1,
                self.atoms,
                inputs.shape[1],
                inputs.shape[2],
            ),
        )
        positive = torch.sum(torch.abs(target_states + input_states), dim=(1, 3))
        negative = torch.sum(torch.abs(target_states - input_states), dim=(1, 3))

        result = torch.min(positive, negative)
        return torch.sum(result) / targets.numel()

    def clone(self) -> "Loss":
        # Make a copy of the loss.
        return deepcopy(self)


class PhaseLossAtomisticMSE(nn.Module):
    """
    A custom loss function for phase loss.
    The loss of each element multiplied by 1 or -1 is taken,
    the lowest value gets returned.
    Basicly a lossless MSE implementation
    """

    def __init__(self, atoms: int = None):
        super().__init__()
        self.atoms = atoms

    def forward(self, inputs, targets) -> torch.Tensor:
        # reshape to(batch, atoms, states, xyz)
        target_states = torch.reshape(
            targets,
            (
                -1,
                self.atoms,
                targets.shape[1],
                targets.shape[2],
            ),
        )
        input_states = torch.reshape(
            inputs,
            (
                -1,
                self.atoms,
                targets.shape[1],
                targets.shape[2],
            ),
        )

        positive = torch.sum(torch.square(target_states + input_states), dim=(1, 3))
        negative = torch.sum(torch.square(target_states - input_states), dim=(1, 3))

        result = torch.min(positive, negative)
        return torch.sum(result) / targets.numel()

    def clone(self) -> "Loss":
        """Make a copy of the loss."""
        return deepcopy(self)

class PhaseLoss(nn.Module):
    """
    The PhaseLoss class is a custom loss function for bulk properties
    that emerge when two distinct electronic states are coupled. 
    It is intended for non-atomistic properties, such as dipoles,
    that have a shape of `(batch_size*n_dipoles, 3)`.

    One example are transition dipoles, which are influenced by the dipole 
    operator :math:`\hat{\mu}` and read as

    .. math::

        \\mu_{ij}(\\mathbf{R}) = \\left\\langle\\Psi_i(\\mathbf{R})|\\hat{\\mu}|\\Psi_j(\\mathbf{R})\\right\\rangle


    As the coupled wavefunctions :math:`\Psi_i` and :math:`\Psi_j` have 
    arbitrary signs, also the resulting property possesses an arbitrary sign.
    The main feature of the customized loss function is that calculates a 
    phase-independent loss. 
    It implements a Mean Square Error (MSE) calculation (default) or Mean Average Error 
    calculation (mse=False). The loss of each element multiplied by 1 or -1 is taken,
    the lowest value gets returned.

    During calculation, the reference data and predictions (targets and
    inputs) are subtracted and added, respectively and all values are
    squared (MSE) or not-squared (MA). The absolute values of these two tensors 
    are computed and summed over the xyz-axis, resulting in two separate tensors:
    a positive tensor and a negative tensor. The minimum value between
    the positive and negative tensors is then computed, and the values
    are summed over all axes and divided by the total number of elements
    in the target.

    The forward() method takes inputs and targets as arguments and re-
    turns a float, i.e., MSE or MAE loss value (L) as the result.

    For dipoles of shape (:math:`N = [1, N_D, 3]`), the PhaseLoss
    is defined as

    .. math::

        \\mathcal{L} = \\frac{1}{3N} \\sum_k^{N_D}\\min_i\\left(\\sum_l^3 | D_k^{ref}1_2^{\\top}
        - D_k^{pred}1_2^{\\top}\\odot \\begin{pmatrix} 1 \\\\ -1 \\end{pmatrix}^{\\top} |_{l}^2\\right)

    """

    def __init__(self, 
                 mse: Optional[str] = True,
            ):

        super().__init__()
        self._mse = mse 

    def forward(self, inputs, targets) -> torch.Tensor:

        if self._mse:
            positive = torch.sum(torch.square(targets + inputs), dim=1)
            negative = torch.sum(torch.square(targets - inputs), dim=1)
        else:
            positive = torch.sum(torch.abs(targets + inputs), dim=1)
            negative = torch.sum(torch.abs(targets - inputs), dim=1)

        result = torch.min(positive, negative)
        return torch.sum(result) / targets.numel()

    def clone(self) -> "Loss":
        """Make a copy of the loss."""
        return deepcopy(self)


class PhaseLossMAE(nn.Module):
    """
    The PhaseLoss class is a custom loss function used to calculate the
    phase loss. It implements a lossless Mean Absolute Error (MAE) cal-
    culation. The loss of each element multiplied by 1 or -1 is taken,
    the lowest value gets returned.

    It is intended for non-atomistic properties, such as dipoles,
    that have a shape of (batch_size, n_dipoles, xyz).

    During calculation, the reference data and predictions (targets and
    inputs) are subtracted and added, respectively. The absolute values
    of these two tensors are computed and summed over the xyz-axis, re-
    sulting in two separate tensors: a positive tensor and a negative
    tensor. The minimum value between the positive and negative tensors
    is then computed, and the values are summed over all axes and
    divided by the total number of elements in the target.

    The forward() method takes inputs and targets as arguments and re-
    turns a float, i.e., MAE loss value (L) as the result.

    For dipoles of shape (N=batch*ND, xyz), the MAE loss is defined as
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets) -> torch.Tensor:
        positive = torch.sum(torch.abs(targets + inputs), dim=1)
        negative = torch.sum(torch.abs(targets - inputs), dim=1)

        result = torch.min(positive, negative)
        return torch.sum(result) / targets.numel()

    def clone(self) -> "Loss":
        """Make a copy of the loss."""
        return deepcopy(self)


class PhaseLossMSE(nn.Module):
    """
    The PhaseLossMSE class is a custom loss function for bulk properties
    that emerge when two distinct electronic states are coupled. 
    It is intended for non-atomistic properties, such as dipoles,
    that have a shape of `(batch_size*n_dipoles, 3)`.

    One example are transition dipoles, which are influenced by the dipole 
    operator :math:`\hat{\mu}` and read as

    .. math::

        \\mu_{ij}(\\mathbf{R}) = \\left\\langle\\Psi_i(\\mathbf{R})|\\hat{\\mu}|\\Psi_j(\\mathbf{R})\\right\\rangle


    As the coupled wavefunctions :math:`\Psi_i` and :math:`\Psi_j` have 
    arbitrary signs, also the resulting property possesses an arbitrary sign.
    The main feature of the customized loss function is that calculates a 
    phase-independent loss. It implements a lossless Mean Square Error (MSE) cal-
    culation. The loss of each element multiplied by 1 or -1 is taken,
    the lowest value gets returned.

    During calculation, the reference data and predictions (targets and
    inputs) are subtracted and added, respectively and all values are
    squared. The absolute values of these two tensors are computed and
    summed over the xyz-axis, resulting in two separate tensors:
    a positive tensor and a negative tensor. The minimum value between
    the positive and negative tensors is then computed, and the values
    are summed over all axes and divided by the total number of elements
    in the target.

    The forward() method takes inputs and targets as arguments and re-
    turns a float, i.e., MSE loss value (L) as the result.

    For dipoles of shape (:math:`N = [1, N_D, 3]`), the PhaseLossMSE
    is defined as

    .. math::

        \\mathcal{L} = \\frac{1}{3N} \\sum_k^{N_D}\\min_i\\left(\\sum_l^3 | D_k^{ref}1_2^{\\top}
        - D_k^{pred}1_2^{\\top}\\odot \\begin{pmatrix} 1 \\\\ -1 \\end{pmatrix}^{\\top} |_{l}^2\\right)

    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets) -> torch.Tensor:
        positive = torch.sum(torch.square(targets + inputs), dim=1)
        negative = torch.sum(torch.square(targets - inputs), dim=1)

        result = torch.min(positive, negative)
        return torch.sum(result) / targets.numel()

    def clone(self) -> "Loss":
        """Make a copy of the loss."""
        return deepcopy(self)


class PhysPhaseLossAtomistic(nn.Module):
    """
    The PhasePropLoss class is a custom loss function used to calculate the
    loss of properties with arbitrary phase. It implements a mean square 
    error (MSE) calculation.

    Loss function is re-implemented from SchNarc [1]

    [1] J. Westermayr, M. Gastegger, and P. Marquetand, J. Phys. Chem. Lett. 2020, 
        11 (10), 3828-3834, 10.1021/acs.jpclett.0c00527 
    """

    def __init__(self, 
            n_states: int,
            n_phases: Optional[int] = None,
            n_couplings: Optional[int] = None,
            mse: Optional[str] = True,
            ):

        super().__init__()

        self.criterion = torch.nn.MSELoss() if mse else torch.nn.L1Loss()

        self.n_states = n_states
        self.n_couplings = int(0.5*n_states*(n_states-1)) if not n_couplings else n_couplings
        self.n_phases = int(2**(n_states-1)) if not n_phases else n_phases
    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.phasemat = self._get_phasematrix(
                n_states=self.n_states, 
                n_phases=self.n_phases, 
                n_couplings=self.n_couplings
                )

        print(self.phasemat.shape, self.phasemat)

    def _get_phasematrix(self, n_states, n_phases, n_couplings):

        phasevec4states = self.__get_phasevector(n_states=n_states, n_phases=n_phases)

        # generate list of states i and j that are coupled
        istates, jstates = np.triu_indices(n_states, k=1)
        e_ik = phasevec4states[istates,:]
        e_jk = phasevec4states[jstates,:]

        n_ci, n_pi = e_ik.shape
        n_cj, n_pj = e_jk.shape

        assert n_pi == n_pj == n_phases, f"Found {n_pj} phases but expect {n_phases}."
        assert n_ci == n_cj == n_couplings, f"Found {n_cj} couplings but expect {n_couplings}."
        
        phasematrix = e_ik * e_jk

        return phasematrix

    def __get_phasevector(self, n_states, n_phases):

        for state in range(n_states):
            sublist = [-1]*state + [1]*(n_states-state)

            # generate permutations for all electronic states
            if state == 0:
                vec = [list(x) for x in list(set(itertools.permutations(sublist, n_states)))]
            else:
                vec.extend([list(x) for x in list(set(itertools.permutations(sublist, n_states)))])

        # apply side condition: phase of first state is always +1, so delete all permutations that have -1
        # as first element
        vec_icond = [p for p in vec if p[0] == 1]

        # check if number of permutations equals number of requested phases
        assert len(vec_icond) == n_phases, f"Found {len(vec_icond)} permutations, but expects {n_phases} phases."

        # reshape list of vectors to array, transpose it and make a torch tensor
        phasevector = torch.as_tensor(np.transpose(np.array(vec_icond)), device=self.device)

        return phasevector

    def forward(self, output, target):

        # get number of couplings (e.g. NACs from shape of data)
        _, n_nacs, _ = output.shape

        # check if shape of data and requested number of NACs is equal
        assert n_nacs == self.n_couplings, f"Found {n_nacs} couplings, but expect {self.n_couplings}."

        # correct predictions by element-wise multiplication with phasematrix
        output_phasecorr = torch.einsum('ijk,jl->jilk', output, self.phasemat)
        # bring targets in same shape as phase-corrected predictions (copy n_phases times)
        target_phase = torch.einsum('ijk,jl->jilk', target, torch.ones(self.phasemat.shape, device=self.device))

        # calculate loss between phasecorrected predictions and targets 
        criterion = self.criterion
        l_loss = [criterion(output_phasecorr[:,:,phase,:], target_phase[:,:,phase,:]) for phase in range(self.n_phases)]
        # keep only the loss of the phasecombination giving the lowest loss value
        loss = min(l_loss)

        return loss


class PhysPhaseLoss(nn.Module):
    """
    The PhasePropLoss class is a custom loss function used to calculate the
    loss of properties with arbitrary phase. It implements a mean square 
    error (MSE) calculation.
    """

    pass


