"""Loss functions for PITS-MRAS (Phase 3).

Holds the composite training objective and its components: physics, temporal,
stability, IRL and HJB losses, aggregated by :class:`TotalLoss`.

``TotalLoss`` combines pre-computed per-component scalar losses using the
weights from :class:`~pits_mras.config.LossConfig` and returns a dict with the
total plus per-component logging scalars (keys ``loss/physics`` etc.).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from pits_mras.config import LossConfig
from pits_mras.losses.hjb import HJBResidualLoss, LyapunovDecreaseEnforcer
from pits_mras.losses.irl import IRLBellmanAccumulator, IRLBellmanLoss
from pits_mras.losses.physics import PhysicsLoss
from pits_mras.losses.stability import (
    ControlEffortLoss,
    LyapunovConstraintLoss,
    MRASStabilityLoss,
    ParameterBoundednessLoss,
)
from pits_mras.losses.temporal import (
    AttentionRegularizationLoss,
    MultiStepPredictionLoss,
    TemporalLoss,
    TemporalSmoothnessLoss,
)

__all__ = [
    "TotalLoss",
    "PhysicsLoss",
    "TemporalLoss",
    "MultiStepPredictionLoss",
    "AttentionRegularizationLoss",
    "TemporalSmoothnessLoss",
    "LyapunovConstraintLoss",
    "ParameterBoundednessLoss",
    "ControlEffortLoss",
    "MRASStabilityLoss",
    "IRLBellmanAccumulator",
    "IRLBellmanLoss",
    "HJBResidualLoss",
    "LyapunovDecreaseEnforcer",
]

# Component name -> (LossConfig attribute, logging key).
_COMPONENTS = {
    "physics": ("lambda_physics", "loss/physics"),
    "temporal": ("lambda_temporal", "loss/temporal"),
    "stability": ("lambda_stability", "loss/stability"),
    "irl": ("lambda_irl", "loss/irl"),
    "hjb": ("lambda_hjb", "loss/hjb"),
    "costate": ("lambda_costate", "loss/costate"),
    "data": ("lambda_data", "loss/data"),
}


class TotalLoss(nn.Module):
    """Weighted sum of the per-component scalar losses.

    ``forward`` takes a mapping from component name (any subset of
    ``physics, temporal, stability, irl, hjb, costate, data``) to a scalar
    loss tensor and returns ``{"loss": total, "loss/<name>": weighted, ...}``.
    Missing components are treated as zero.
    """

    def __init__(self, config: LossConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else LossConfig()

    def forward(self, components: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        total: torch.Tensor | None = None
        out: dict[str, torch.Tensor] = {}
        for name, (attr, log_key) in _COMPONENTS.items():
            if name not in components:
                continue
            weight = getattr(self.config, attr)
            weighted = weight * components[name]
            out[log_key] = weighted
            total = weighted if total is None else total + weighted
        if total is None:
            total = torch.zeros(())
        out["loss"] = total
        return out
