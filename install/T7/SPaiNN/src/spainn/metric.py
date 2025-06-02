from typing import Tuple

import torch
from torchmetrics import Metric

__all__ = ["PhaseMAE", "PhaseAtomisticMAE", "PhaseAtomisticMSE"]


class PhaseMAE(Metric):
    """
    Metric to monitor the phaseless loss of bulk properties obtained by
    means of the Mean Absolute Error (MAE).
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    sum_abs_error: torch.Tensor
    total: torch.Tensor

    def __init__(self):
        super().__init__()
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        sum_abs_error, n_obs = self._mae_update(preds, target)

        self.sum_abs_error += sum_abs_error
        self.total += n_obs

    def compute(self) -> torch.Tensor:
        return self.sum_abs_error / self.total

    def _mae_update(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.tensor, int]:
        assert preds.shape == target.shape

        preds = preds if preds.is_floating_point else preds.float()
        target = target if target.is_floating_point else target.float()
        positive = torch.sum(torch.abs(target + preds), dim=1)
        negative = torch.sum(torch.abs(target - preds), dim=1)
        sum_abs_error = torch.sum(torch.min(positive, negative))
        n_obs = target.numel()

        return sum_abs_error, n_obs


class PhaseAtomisticMAE(Metric):
    """
    Metric to monitor the phaseless loss of atomistic properties obtained by
    means of the Mean Absolute Error (MAE).
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    sum_abs_error: torch.Tensor
    total: torch.Tensor

    def __init__(self, atoms: int):
        super().__init__()
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.atoms = atoms

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        sum_abs_error, n_obs = self._mae_update(preds, target)

        self.sum_abs_error += sum_abs_error
        self.total += n_obs

    def compute(self) -> torch.Tensor:
        return self.sum_abs_error / self.total

    def _mae_update(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.tensor, int]:
        assert preds.shape == target.shape

        preds = preds if preds.is_floating_point else preds.float()
        target = target if target.is_floating_point else target.float()
        target_states = torch.reshape(
            target,
            (
                -1,
                self.atoms,
                target.shape[1],
                target.shape[2],
            ),
        )
        input_states = torch.reshape(
            preds,
            (
                -1,
                self.atoms,
                preds.shape[1],
                preds.shape[2],
            ),
        )

        positive = torch.sum(torch.abs(target_states + input_states), dim=(1, 3))
        negative = torch.sum(torch.abs(target_states - input_states), dim=(1, 3))
        sum_abs_error = torch.sum(torch.min(positive, negative))
        n_obs = target.numel()

        return sum_abs_error, n_obs


class PhaseAtomisticMSE(Metric):
    """
    Metric to monitor the phaseless loss of atomistic properties obtained by
    means of the Mean Squared Error (MSE).
    """

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    sum_abs_error: torch.Tensor
    total: torch.Tensor

    def __init__(self, atoms: int):
        super().__init__()
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.atoms = atoms

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        sum_abs_error, n_obs = self._mae_update(preds, target)

        self.sum_abs_error += sum_abs_error
        self.total += n_obs

    def compute(self) -> torch.Tensor:
        return self.sum_abs_error / self.total

    def _mae_update(
        self, preds: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.tensor, int]:
        assert preds.shape == target.shape

        preds = preds if preds.is_floating_point else preds.float()
        target = target if target.is_floating_point else target.float()
        target_states = torch.reshape(
            target,
            (
                -1,
                self.atoms,
                target.shape[1],
                target.shape[2],
            ),
        )
        input_states = torch.reshape(
            preds,
            (
                -1,
                self.atoms,
                preds.shape[1],
                preds.shape[2],
            ),
        )

        positive = torch.sum(torch.square(target_states + input_states), dim=(1, 3))
        negative = torch.sum(torch.square(target_states - input_states), dim=(1, 3))

        sum_abs_error = torch.sum(torch.min(positive, negative))
        n_obs = target.numel()

        return sum_abs_error, n_obs

