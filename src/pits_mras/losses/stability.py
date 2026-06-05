"""Stability-constraint losses (Phase 3).

Grounded in §3.3 / §3.5: a valid Lyapunov function must satisfy V̇ < 0 along
trajectories.  ``LyapunovConstraintLoss`` penalizes the violation
``ReLU(V̇ + margin)``; ``ParameterBoundednessLoss`` keeps adaptive parameters
bounded (L2); ``ControlEffortLoss`` penalizes ``uᵀRu`` (the control quadratic
form from §3.1); ``MRASStabilityLoss`` aggregates the three.

NOTE (reasonable-minimal): the impl plan §6 does not spell out closed-form
stability sub-losses, so the standard mathematically-correct forms are used
and the aggregator weights default to 1.0 (tunable).
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class LyapunovConstraintLoss(nn.Module):
    """Penalize ``V̇ > -margin`` via ``mean(ReLU(V̇ + margin))``.

    ``vdot`` is the Lyapunov derivative ``V̇`` (shape ``[batch]`` or any shape);
    a positive ``margin`` enforces a strict decrease rate ``V̇ < -margin``.
    """

    def __init__(self, margin: float = 0.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, vdot: torch.Tensor) -> torch.Tensor:
        return torch.relu(vdot + self.margin).mean()


class ParameterBoundednessLoss(nn.Module):
    """Sum of squared L2 norms of the supplied adaptive parameters."""

    def forward(self, params: Iterable[torch.Tensor]) -> torch.Tensor:
        params = list(params)
        total = torch.zeros((), device=params[0].device, dtype=params[0].dtype)
        for p in params:
            total = total + (p**2).sum()
        return total


class ControlEffortLoss(nn.Module):
    """Quadratic control-effort penalty ``E[ uᵀ R u ]`` (§3.1)."""

    R: torch.Tensor

    def __init__(self, R: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("R", R)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        uRu = torch.einsum("bi,ij,bj->b", u, self.R, u)
        return uRu.mean()


class MRASStabilityLoss(nn.Module):
    """Aggregate Lyapunov-decrease, parameter-boundedness and control-effort."""

    def __init__(
        self,
        R: torch.Tensor,
        margin: float = 0.0,
        lambda_lyapunov: float = 1.0,
        lambda_param: float = 1.0,
        lambda_effort: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_lyapunov = lambda_lyapunov
        self.lambda_param = lambda_param
        self.lambda_effort = lambda_effort
        self.lyapunov = LyapunovConstraintLoss(margin)
        self.param = ParameterBoundednessLoss()
        self.effort = ControlEffortLoss(R)

    def forward(
        self,
        vdot: torch.Tensor,
        u: torch.Tensor,
        params: Iterable[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        loss_lyap = self.lyapunov(vdot)
        loss_param = self.param(params)
        loss_effort = self.effort(u)
        loss = (
            self.lambda_lyapunov * loss_lyap
            + self.lambda_param * loss_param
            + self.lambda_effort * loss_effort
        )
        return {
            "loss": loss,
            "lyapunov": loss_lyap,
            "param": loss_param,
            "effort": loss_effort,
        }
