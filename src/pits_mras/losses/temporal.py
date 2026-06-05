"""Temporal / multi-step prediction loss (Phase 3).

``TemporalLoss`` implements §6.2 (multi-step prediction with optional
attention weighting and a smoothness penalty).  The ROADMAP-named
``MultiStepPredictionLoss``, ``AttentionRegularizationLoss`` and
``TemporalSmoothnessLoss`` are composable building blocks usable on their own.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiStepPredictionLoss(nn.Module):
    """Mean squared multi-step prediction error.

    ``predictions``/``targets``: ``[batch, horizon, state_dim]``.  Optional
    ``attention_weights`` ``[batch, horizon]`` (should sum to 1 over horizon)
    re-weights the per-step error; otherwise the horizon is averaged.
    """

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        err = predictions - targets
        sq = (err**2).sum(dim=-1)  # [batch, horizon]
        if attention_weights is not None:
            weighted = (sq * attention_weights).sum(dim=-1)  # [batch]
        else:
            weighted = sq.mean(dim=-1)  # [batch]
        return weighted.mean()


class TemporalSmoothnessLoss(nn.Module):
    """Penalize large step-to-step changes in the predicted trajectory."""

    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        dpred = predictions[:, 1:, :] - predictions[:, :-1, :]
        return (dpred**2).sum(dim=-1).mean()


class AttentionRegularizationLoss(nn.Module):
    """Negative-entropy regularizer on attention weights.

    Returns ``E[ Σ w·log w ]`` (negative entropy) over the last axis, so
    peaked (low-entropy) distributions incur a *larger* penalty than uniform
    ones.  Useful to discourage attention collapse onto a single step.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, attention_weights: torch.Tensor) -> torch.Tensor:
        w = attention_weights.clamp_min(self.eps)
        neg_entropy = (w * torch.log(w)).sum(dim=-1)
        return neg_entropy.mean()


class TemporalLoss(nn.Module):
    """Multi-step prediction loss with optional attention weighting (§6.2)."""

    def __init__(self, horizon: int = 5, lambda_smooth: float = 0.0) -> None:
        super().__init__()
        self.horizon = horizon
        self.lambda_smooth = lambda_smooth
        self._prediction = MultiStepPredictionLoss()
        self._smoothness = TemporalSmoothnessLoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        loss_pred = self._prediction(predictions, targets, attention_weights)

        loss_smooth = torch.zeros((), device=predictions.device, dtype=predictions.dtype)
        if self.lambda_smooth > 0:
            loss_smooth = self._smoothness(predictions)

        loss = loss_pred + self.lambda_smooth * loss_smooth

        return {
            "loss": loss,
            "prediction": loss_pred,
            "smoothness": loss_smooth,
        }
