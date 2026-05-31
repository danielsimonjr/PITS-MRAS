r"""HJB residual loss (Phase 3).

Implements §3.5:

    L_HJB = ‖ eᵀQe + (u*)ᵀRu* + ∇_eV̂·(A_m e + B u* + f_corr) ‖²

The default HJB weight is small (λ_HJB = 0.01): per the impl plan it is treated
as a tunable regularizer (HJBPPO found no consistent gain).

``LyapunovDecreaseEnforcer`` is the model-based decrease constraint
``ReLU(∇V̂·f + ℓ)`` (§3.3/§3.5 intent): it penalizes directions in which the
value function fails to decrease along the dynamics ``f``.

FACTOR-OF-½ RECONCILIATION (load-bearing -- flagged ADR).  §3.3 writes the
optimal control as ``u* = −R⁻¹Bᵀ∇_eV̂`` AND (line 142) claims this equals the
LQR gain ``u* = −R⁻¹BᵀPe = −Ke``.  Both cannot hold simultaneously: this
critic stores ``V̂ = eᵀPe`` so ``∇_eV̂ = 2Pe``, and the *un-halved* formula
gives ``u* = −2Ke`` -- which makes the HJB residual non-zero at the CARE
optimum.  The mathematically-consistent optimal control for ``V̂ = eᵀPe`` is

    u* = −½ R⁻¹ Bᵀ ∇_eV̂  =  −R⁻¹ Bᵀ P e  =  −K e,

which is exactly the LQR gain the spec claims AND drives the HJB residual to
zero at the CARE solution.  We therefore implement the ½-scaled form (ROADMAP
§6 "factor-of-½ convention" caveat).  ``half_grad=False`` recovers the literal
un-halved §3.5 text for callers who store ``V̂ = ½eᵀPe`` instead.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from pits_mras.models.critic import QuadraticCritic


class HJBResidualLoss(nn.Module):
    """Hamilton-Jacobi-Bellman residual loss (§3.5)."""

    A_m: torch.Tensor
    B: torch.Tensor
    Q: torch.Tensor
    R: torch.Tensor
    R_inv: torch.Tensor

    def __init__(
        self,
        A_m: torch.Tensor,
        B: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        weight: float = 0.01,
        half_grad: bool = True,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.half_grad = half_grad
        self.register_buffer("A_m", A_m)
        self.register_buffer("B", B)
        self.register_buffer("Q", Q)
        self.register_buffer("R", R)
        self.register_buffer("R_inv", torch.linalg.inv(R))

    def forward(
        self,
        critic: QuadraticCritic,
        e: torch.Tensor,
        f_corr: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        grad_v: torch.Tensor = critic.gradient(e)  # ∇_e V̂  [batch, state_dim]
        # u* = −(½) R⁻¹ Bᵀ ∇_e V̂  (½ for V̂=eᵀPe -> u*=−Ke; see module doc).
        scale = 0.5 if self.half_grad else 1.0
        bt_grad = torch.einsum("ij,bi->bj", self.B, grad_v)  # Bᵀ∇V̂ [batch, m]
        u_star = -scale * torch.einsum("ij,bj->bi", self.R_inv, bt_grad)

        eQe = torch.einsum("bi,ij,bj->b", e, self.Q, e)
        uRu = torch.einsum("bi,ij,bj->b", u_star, self.R, u_star)

        drift = torch.einsum("ij,bj->bi", self.A_m, e) + torch.einsum(
            "ij,bj->bi", self.B, u_star
        )
        if f_corr is not None:
            drift = drift + f_corr
        grad_dot_f = (grad_v * drift).sum(dim=-1)

        residual = eQe + uRu + grad_dot_f
        loss = self.weight * (residual ** 2).mean()
        return {"loss": loss, "residual": residual, "u_star": u_star}


class LyapunovDecreaseEnforcer(nn.Module):
    """Penalize ``∇V̂·f + ℓ > 0`` via ``mean(ReLU(∇V̂·f + margin))``.

    ``grad_v`` and ``f`` are ``[batch, state_dim]``; the dot product is the
    Lyapunov derivative along the dynamics ``f``.  A positive ``margin`` (ℓ)
    enforces a strict decrease.
    """

    def __init__(self, margin: float = 0.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, grad_v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        vdot = (grad_v * f).sum(dim=-1)
        return torch.relu(vdot + self.margin).mean()
