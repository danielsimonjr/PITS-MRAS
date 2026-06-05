"""Inverse-RL Bellman residual loss (Phase 3).

Implements the load-bearing IRL reward-consistency residual from §3.2:

    δ_IRL(t) = ∫_{t−T}^{t}(eᵀQe + uᵀRu)dτ − [V̂(e(t−T)) − V̂(e(t))]
    L_IRL = ½ · E[ δ_IRL² ]

The IRL Bellman equation is *model-free*: the drift matrix ``A`` does NOT
appear.  ``IRLBellmanAccumulator`` computes the trapezoidal running-cost
integral over a trajectory window; ``IRLBellmanLoss`` forms the residual
against a critic's value difference and returns ``½·E[δ²]``.

NOTE on the value-difference sign: the integral form of the Bellman equation
(impl plan §6.4, line 1090) reads ``V(t−T) = ∫_{t−T}^{t} r dτ + V(t)``, i.e.
``∫ r dτ = V(t−T) − V(t)``.  The residual that vanishes at the true value is
therefore ``δ = ∫ r dτ − [V̂(t−T) − V̂(t)]``.  The §3.2 line writes the bracket
as ``[V̂(t) − V̂(t−T)]`` (reversed); that ordering does *not* vanish at the true
value (it gives ``2(V(t−T) − V(t))``).  We follow the integral Bellman equation
(the correctness contract) so the load-bearing "residual ≈ 0 at the true value"
property holds.  The squared loss is sign-insensitive regardless.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pits_mras.models.critic import QuadraticCritic


class IRLBellmanAccumulator(nn.Module):
    """Trapezoidal accumulator for ∫(eᵀQe + uᵀRu)dτ over a window.

    Given a trajectory window ``e`` of shape ``[batch, T+1, state_dim]`` and a
    control window ``u`` of shape ``[batch, T+1, action_dim]`` sampled at a
    uniform step ``dt``, returns the integrated running cost per batch element
    with shape ``[batch]``.
    """

    Q: torch.Tensor
    R: torch.Tensor

    def __init__(self, Q: torch.Tensor, R: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("Q", Q)
        self.register_buffer("R", R)

    def forward(self, e: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        # Running cost density g(τ) = eᵀQe + uᵀRu, shape [batch, T+1].
        eQe = torch.einsum("bti,ij,btj->bt", e, self.Q, e)
        uRu = torch.einsum("bti,ij,btj->bt", u, self.R, u)
        g = eQe + uRu  # [batch, T+1]
        # Composite trapezoidal rule over the time axis.
        integral = dt * (g[:, 1:] + g[:, :-1]).sum(dim=-1) * 0.5
        return integral


class IRLBellmanLoss(nn.Module):
    """IRL Bellman residual loss  L_IRL = ½ · E[ δ_IRL² ]  (§3.2)."""

    def __init__(self, Q: torch.Tensor, R: torch.Tensor) -> None:
        super().__init__()
        self.accumulator = IRLBellmanAccumulator(Q, R)

    def forward(
        self,
        critic: QuadraticCritic,
        e: torch.Tensor,
        u: torch.Tensor,
        dt: float,
    ) -> dict[str, torch.Tensor]:
        integral = self.accumulator(e, u, dt)  # [batch]
        v_end = critic(e[:, -1, :])  # V̂(e(t))    [batch]
        v_start = critic(e[:, 0, :])  # V̂(e(t−T))  [batch]
        # ∫ r dτ = V(t−T) − V(t)  =>  δ = ∫ − [V̂(t−T) − V̂(t)]  (see module doc).
        delta = integral - (v_start - v_end)  # [batch]
        loss = 0.5 * (delta**2).mean()
        return {"loss": loss, "delta": delta}
