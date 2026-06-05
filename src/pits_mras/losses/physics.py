"""Physics-informed loss (Phase 3).

Implements §6.1 verbatim:

    energy_residual = dH_dt − (P_control − P_diss)

with λ-weighted energy / PDE / boundary-condition / symmetry residual terms.
The PDE, BC and symmetry residuals are optional (default ``None`` -> zero).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PhysicsLoss(nn.Module):
    """Energy-conservation + PDE/BC/symmetry residual loss.

    energy_residual = dH_dt − (P_control − P_diss)
    """

    def __init__(
        self,
        lambda_energy: float = 1.0,
        lambda_pde: float = 1.0,
        lambda_bc: float = 1.0,
        lambda_sym: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_energy = lambda_energy
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_sym = lambda_sym

    def forward(
        self,
        dH_dt: torch.Tensor,
        P_control: torch.Tensor,
        P_diss: torch.Tensor,
        pde_residual: torch.Tensor | None = None,
        bc_residual: torch.Tensor | None = None,
        sym_residual: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        energy_residual = dH_dt - (P_control - P_diss)
        loss_energy = (energy_residual**2).mean()

        zero = torch.zeros((), device=dH_dt.device, dtype=dH_dt.dtype)
        loss_pde = (pde_residual**2).mean() if pde_residual is not None else zero
        loss_bc = (bc_residual**2).mean() if bc_residual is not None else zero
        loss_sym = (sym_residual**2).mean() if sym_residual is not None else zero

        loss = (
            self.lambda_energy * loss_energy
            + self.lambda_pde * loss_pde
            + self.lambda_bc * loss_bc
            + self.lambda_sym * loss_sym
        )

        return {
            "loss": loss,
            "energy": loss_energy,
            "pde": loss_pde,
            "bc": loss_bc,
            "sym": loss_sym,
        }
