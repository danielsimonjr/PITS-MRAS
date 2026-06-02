r"""Physics-Constrained Machine Learning (PCML) module for PITS-MRAS.

Implements two constraint-enforcement modes that upgrade the soft, port-
Hamiltonian physics regularizer into provable constraint satisfaction:

1. **Soft PCML** (Patel et al., IFAC 2022): augment the loss with constraint
   residuals, ``L = lambda_diff*||D||^2 + lambda_eq*||h||^2 + lambda_ineq*||ReLU(g)||^2``.
   Reduces violations probabilistically; used in early training.
2. **Hard PCML** (DAE-HardNet, Golder et al., arXiv:2512.05881): project the
   network output onto the DAE constraint manifold by solving the KKT system of
   a minimum-distance problem with a differentiable Newton solver, achieving
   point-wise constraint satisfaction. Activated dynamically once the backbone
   data loss drops below ``eta``.

The :class:`PCMLModule` wraps a backbone's prediction ``f_hat`` and returns the
constrained prediction plus the PCML loss, switching between modes by the
``eta`` threshold (DAE-HardNet dynamic activation).

References:
- Patel, Bhartiya & Gudi, *Physics Constrained Learning in NN based Modeling*,
  IFAC-PapersOnLine 55-7 (2022) 79-85. [soft]
- Golder, Roy & Hasan, *DAE-HardNet*, arXiv:2512.05881 (2025). [hard]
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pits_mras.constraints.base import PhysicsConstraints


class SoftPCMLLoss(nn.Module):
    r"""Soft physics-constraint loss (Patel et al. 2022, Eqs. 3-4).

    ``L = lambda_diff*||D(x,t,y,d)||^2 + lambda_eq*||h(x,t,y)||^2
          + lambda_ineq*||ReLU(g(x,t,y))||^2``

    This generalizes the existing port-Hamiltonian ``L_physics`` (energy / PDE /
    BC residuals are special cases of the differential / equality residuals).
    """

    def __init__(
        self,
        constraints: PhysicsConstraints,
        lambda_diff: float = 1.0,
        lambda_eq: float = 1.0,
        lambda_ineq: float = 0.5,
    ) -> None:
        super().__init__()
        self.constraints = constraints
        self.lambda_diff = lambda_diff
        self.lambda_eq = lambda_eq
        self.lambda_ineq = lambda_ineq

    def forward(
        self, x: Tensor, t: Tensor, y_pred: Tensor, d_pred: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Return ``(total, breakdown)`` with keys ``diff``, ``eq``, ``ineq``, ``violation``."""
        diff = self.constraints.differential(x, t, y_pred, d_pred)
        eq = self.constraints.equality(x, t, y_pred)
        ineq = self.constraints.inequality(x, t, y_pred)

        zero = y_pred.new_zeros(())
        l_diff = (diff**2).mean() if diff.numel() > 0 else zero
        l_eq = (eq**2).mean() if eq.numel() > 0 else zero
        l_ineq = (F.relu(ineq) ** 2).mean() if ineq.numel() > 0 else zero

        total = (
            self.lambda_diff * l_diff
            + self.lambda_eq * l_eq
            + self.lambda_ineq * l_ineq
        )
        violation = self.constraints.violation(x, t, y_pred, d_pred)
        return total, {
            "diff": l_diff,
            "eq": l_eq,
            "ineq": l_ineq,
            "violation": violation,
        }


class TaylorNeighborhoodApproximation(nn.Module):
    r"""Multi-point neighborhood approximation (DAE-HardNet §3, Eq. 9).

    Expresses ``y(x, t)`` as a weighted combination of its values at neighbor
    points minus the derivative corrections, converting differential operators
    into algebraic variables ``d`` for the KKT projection::

        y ~ (1/|X|) sum_i [ y([x,t] + Delta_i) - Delta * d_i - 0.5 * Delta^2 * d_ii ]

    where ``|X| = input_dim`` (one neighbor per independent variable). The
    approximation error vanishes as ``Delta -> 0`` (recommended ``Delta`` in
    ``[1e-3, 0.1]``; too small risks an ill-conditioned KKT Jacobian).
    """

    def __init__(
        self,
        backbone: nn.Module,
        input_dim: int,
        delta: float = 0.01,
        order: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_dim = input_dim
        self.delta = delta
        self.order = order

    def forward(self, inputs: Tensor, derivatives: Tensor) -> Tensor:
        """Return the neighborhood approximation ``[batch, output_dim]``.

        ``inputs``: ``[batch, input_dim]`` (the ``(x, t)`` point). ``derivatives``:
        first-order ``d_i`` in the first ``input_dim`` columns and (if
        ``order >= 2``) second-order ``d_ii`` in the next ``input_dim`` columns.
        """
        neighbor_terms = []
        for i in range(self.input_dim):
            delta_vec = torch.zeros_like(inputs)
            delta_vec[:, i] = self.delta
            y_neighbor = self.backbone(inputs + delta_vec)
            d_i = derivatives[:, i : i + 1]
            term = y_neighbor - self.delta * d_i.expand_as(y_neighbor)
            if self.order >= 2:
                d_ii = derivatives[:, self.input_dim + i : self.input_dim + i + 1]
                term = term - 0.5 * (self.delta**2) * d_ii.expand_as(y_neighbor)
            neighbor_terms.append(term)
        return torch.stack(neighbor_terms, dim=0).mean(dim=0)
