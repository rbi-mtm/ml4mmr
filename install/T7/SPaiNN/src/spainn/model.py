from typing import Callable, Dict, Optional, Sequence, Union

import schnetpack as spk
import schnetpack.nn as snn
import torch
import torch.nn as nn
import torch.nn.functional as F
from schnetpack import properties

import spainn
from spainn.properties import SPAINN

__all__ = ["Atomwise", "Nacs", "Forces", "Dipoles", "Socs"]


class Atomwise(nn.Module):
    """
    Predicts atom-wise contributions and accumulates global prediction, *e.g.*, for energies.

    If `aggregation_mode` is None, only the per-atom predictions will be returned.

    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        aggregation_mode: str = "sum",
        output_key: str = SPAINN.energy,
        per_atom_output_key: Optional[str] = None,
        ):

        """
        Parameters
        ----------

        n_in: input dimension of representation
        n_out: output dimension of target property (default: 1), i.e., number of states
        n_hidden: size of hidden layers.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers: number of layers.
        aggregation_mode: one of {sum, avg} (default: sum)
        output_key: the key under which the result will be stored
        per_atom_output_key: If not None, the key under which the per-atom result will be stored

        Returns
        -------

        Dictionary with predicted energies for multiple states.

        Examples
        --------

        The following code demonstrates how to define a prediction module for predicting the
        energies of two electronic states using the :py:class:`Atomwise` module **SPaiNN**.

        >>> pred_energy = spainn.Atomwise(
        >>>     n_in=30,
        >>>     n_out=2
        >>> )

        """

        super().__init__()
        self.output_key = output_key
        self.model_outputs = [output_key]
        self.per_atom_output_key = per_atom_output_key
        self.n_out = n_out

        if aggregation_mode is None and self.per_atom_output_key is None:
            raise ValueError(
                "If `aggregation_mode` is None, `per_atom_output_key` needs to be set,"
                + " since no accumulated output will be returned!"
            )

        self.outnet = spk.nn.build_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )
        self.aggregation_mode = aggregation_mode

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # predict atomwise contributions
        y = self.outnet(inputs["scalar_representation"])

        # aggregate the per-atom output
        if self.aggregation_mode is not None:
            idx_m = inputs[properties.idx_m]
            maxm = int(idx_m[-1]) + 1
            y = snn.scatter_add(y, idx_m, dim_size=maxm)
            y = torch.squeeze(y, -1)

            if self.aggregation_mode == "avg":
                for i in range(len(y[0])):
                    y[:, i] = y[:, i] / inputs[properties.n_atoms]

        inputs[self.output_key] = y
        return inputs

class Nacs(nn.Module):
    """
    Predicts nonadiabatic coupling vector, *i.e.* local, atomic couplings of
    electronic states of the same multiplicity.
    
    This requires a representation supplying (equivariant) vector features [#painnC]_.

    References
    ----------

    .. [#painnC] Schütt, Unke, Gastegger.
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html

    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        nac_key: str = SPAINN.nacs,
        use_vector_repr: bool = False,
    ):
        """
        Parameters
        -----------

        n_in: input dimension of representation
        n_out: output dimension of target property, *i.e.*,
                number of couplings
        n_hidden: size of hidden layers
            If an integer, same number of node is used for all hidden layers
            resulting in a rectangular network.
            If None, the number of neurons is divided by two after each layer
            starting n_in resulting in a pyramidal network.
        n_layers: number of layers
        activation: activation function
        nac_key: the key under which the nonadiabatic couplings are stored
        use_vector_repr: If true, use vector representation to predict
            local, atomic couplings.

        Returns
        --------

        Dictionary with predicted energies for multiple states.

        Examples
        --------

        The following code demonstrates how to define a prediction module for predicting the
        nonadiabatic couplings of two electronic states, *i.e.* one coupling vector using 
        the :py:class:`Nacs` module **SPaiNN**.

        >>> pred_nacs = spainn.Nacs(
        >>>     n_in=30,
        >>>     n_out=1
        >>> )

        """

        super().__init__()
        self.nac_key = nac_key
        self.model_outputs = [nac_key]
        self.use_vector_repr = use_vector_repr

        if self.use_vector_repr:
            self.outnet = spk.nn.build_gated_equivariant_mlp(
                n_in=n_in,
                n_out=n_out,
                n_hidden=n_hidden,
                n_layers=n_layers,
                activation=activation,
                sactivation=activation,
            )
        else:
            self.outnet = spk.nn.build_mlp(
                n_in=n_in,
                n_out=n_out,
                n_hidden=n_hidden,
                n_layers=n_layers,
                activation=activation,
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        l0 = inputs["scalar_representation"]

        if self.use_vector_repr:
            l1 = inputs["vector_representation"]
            nacss, nacsv = self.outnet((l0, l1))

            # Multiply scalar part with positions and add vector part
            # Seems to be the best way to include the scalar part and improve predictions
            nacs_painn = torch.einsum('ij,ik->ijk', inputs[properties.R], nacss) + nacsv
            inputs[self.nac_key] = torch.transpose(nacs_painn, 2, 1).contiguous()

        else:
            virt = self.outnet(l0)
            nacs = spk.nn.derivative_from_molecular(
                virt, inputs["_positions"], True, True
            )[:]
            inputs[self.nac_key] = nacs

        return inputs


class Forces(nn.Module):
    """
    Predicts forces and stress as response of the energy prediction
    w.r.t. the atom positions and strain for multiple electronic states.

    The number of electronic states, for which the forces are computed
    is automatically taken from the shape of the energies. 
    For example, energy inputs of shape :math:`[1,3]` result in the 
    prediction of forces for three electronic states.
    """

    def __init__(
        self,
        calc_forces: bool = True,
        calc_stress: bool = False,
        energy_key: str = SPAINN.energy,
        force_key: str = SPAINN.forces,
    ):
        """
        Parameters
        -----------

        calc_forces: If True, calculate atomic forces.
        calc_stress: If True, calculate the stress tensor.
        energy_key: Key of the energy in results.
        force_key: Key of the forces in results.

        Returns
        --------

        Dictionary with predicted forces for multiple states.

        Examples
        --------

        The following code demonstrates how to define a prediction module for predicting the
        forces from for multiple electronic states from the first derivative of the respective
        predicted energies with respect to the atom positions using the :py:class:`Forces` 
        module.
        The number of the electronic states is automatically taken from the shape of the
        energies.

        >>> pred_forces = spainn.Forces()

        """

        super().__init__()
        self.calc_forces = calc_forces
        self.calc_stress = calc_stress
        self.energy_key = energy_key
        self.force_key = force_key
        self.model_outputs = [force_key]

        self.required_derivatives = []
        if self.calc_forces:
            self.required_derivatives.append(properties.R)
        if self.calc_stress:
            self.required_derivatives.append(properties.strain)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        E_pred = inputs[self.energy_key]
        grads = spk.nn.derivative_from_molecular(
            E_pred, inputs[properties.R], self.training, True
        )

        inputs[self.force_key] = -grads
        return inputs


class Dipoles(nn.Module):
    """
    Predict Dipole Moments from latent partial charges. This requires a
    representation supplying (equivariant) vector features [#painnA]_, [#dipole]_, [#irspec]_.

    References
    ----------

    .. [#painnA] Schütt, Unke, Gastegger.
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html
    .. [#dipole] Veit et al.
       Predicting molecular dipole moments by combining atomic partial charges and atomic dipoles.
       The Journal of Chemical Physics 153.2 (2020): 024113.
    .. [#irspec] Gastegger, Behler, Marquetand.
       Machine learning molecular dynamics for the simulation of infrared spectra.
       Chemical science 8.10 (2017): 6924-6935.

    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        dipole_key: str = SPAINN.dipoles,
        return_charges: bool = True,
        charges_key: str = properties.partial_charges,
        use_vector_repr: bool = False,
    ):
        """
        Parameters
        ----------

        n_in: Input dimension of representation.
        n_out: Output dimension, *i.e.*, number of dipole moments (couplings).
        n_hidden: Size of hidden layers.
            If an integer, same number of node is used for all hidden layers
            resulting in a rectangular network.
            If None, the number of neurons is divided by two after each layer
            starting n_in resulting in a pyramidal network.
        n_layers: Number of layers.
        activation: Activation function.
        dipole_key: Key under which dipoles are stored.
        return_charges: If true, return partial charges.
        charges_key: Key under which partial charges are stored.
        use_vector_repr: If true, use (equivariant) vector representation to 
            dipole momemnts.

        Returns
        --------

        Dictionary with predicted dipoles for multiple states.

        Examples
        --------

        The following code demonstrates how to define a prediction module for predicting the
        dipole moments of two electronic states, *i.e.* one transition dipole and two permanent
        dipoles using the :py:class:`Dipoles` module of **SPaiNN**.

        >>> pred_dipoles = spainn.Dipoles(
        >>>     n_in=30,
        >>>     n_out=1
        >>> )

        """

        super().__init__()
        self.dipole_key = dipole_key
        self.charges_key = charges_key
        self.return_charges = return_charges
        self.model_outputs = [dipole_key]
        self.use_vector_repr = use_vector_repr
        if self.return_charges:
            self.model_outputs.append(self.charges_key)

        if self.use_vector_repr:
            self.outnet = spk.nn.build_gated_equivariant_mlp(
                n_in=n_in,
                n_out=n_out,
                n_hidden=n_hidden,
                n_layers=n_layers,
                activation=activation,
                sactivation=activation,
            )
        else:
            self.outnet = spk.nn.build_mlp(
                n_in=n_in,
                n_out=n_out,
                n_hidden=n_hidden,
                n_layers=n_layers,
                activation=activation,
            )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        positions = inputs[spk.properties.R]
        l0 = inputs["scalar_representation"]
        natoms = inputs[properties.n_atoms]
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1

        if self.use_vector_repr:
            l1 = inputs["vector_representation"]
            charges, atomic_dipoles = self.outnet((l0, l1))
        else:
            charges = self.outnet(l0)
            atomic_dipoles = 0.0

        if properties.total_charge in inputs:
            total_charge = inputs[properties.total_charge]
            sum_charge = snn.scatter_add(charges, idx_m, dim_size=maxm)
            charge_correction = (total_charge[:, None] - sum_charge) / natoms.unsqueeze(
                -1
            )
            charge_correction = charge_correction[idx_m]
            charges = charges + charge_correction

        y = torch.einsum("ij,ik->ijk", positions, charges) + atomic_dipoles
        y = snn.scatter_add(y, idx_m, dim_size=maxm)
        y = torch.transpose(y, 2, 1)
        inputs[self.dipole_key] = y.reshape(-1, 3)

        if self.return_charges:
            inputs[self.charges_key] = charges

        return inputs

class Socs(nn.Module):
    """
    Predicts spin-orbit coupling vectors, *i.e.*, local atomic couplings
    between electronic states of different multiplicity.

    This requires a representation supplying (equivariant) vector features [#painnB]_.

    References
    ----------

    .. [#painnB] Schütt, Unke, Gastegger.
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html

    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        n_hidden: Optional[Union[int, Sequence[int]]] = None,
        n_layers: int = 2,
        activation: Callable = F.silu,
        socs_key: str = SPAINN.socs,
    ):
        """
        Parameters
        ----------
        
        n_in: Input dimension of representation.
        n_out: Output dimension, *i.e.*, number of spin-orbit couplings.
        n_hidden: Size of hidden layers.
            If an integer, same number of node is used for all hidden layers
            resulting in a rectangular network.
            If None, the number of neurons is divided by two after each layer
            starting n_in resulting in a pyramidal network.
        n_layers: Number of layers.
        activation: Activation function.
        socs_key: Key under which spin-orbit couplings are stored.

        Returns
        --------

        Dictionary with predicted dipoles for multiple states.
        
        Examples
        --------

        The following code demonstrates how to define a prediction module for predicting the
        spin-orbit couplings of two electronic states, *e.g.* :math:`S_0` and :math:`T_1` 
        using the :py:class:`Socs` module of **SPaiNN**.

        >>> pred_dipoles = spainn.Dipoles(
        >>>     n_in=30,
        >>>     n_out=1
        >>> )

        """

        super().__init__()
        self.socs_key = socs_key
        self.model_outputs = [socs_key]
        self.n_out = n_out

        self.outnet = spk.nn.build_mlp(
            n_in=n_in,
            n_out=n_out,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=activation,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # predict atomwise contributions
        idx_m = inputs[properties.idx_m]
        maxm = int(idx_m[-1]) + 1
        y = self.outnet(inputs["scalar_representation"])
        y = snn.scatter_add(y, idx_m, dim_size=maxm)
        inputs[self.socs_key] = y
        return inputs
