r"""Actor-critic MRAS controller (IP §7.3). Identities 1, 2, 3, 4.

Owning phase: Phase 4 (Controllers).

Classical MRAS structure::

    u(t) = -K_fb e(t) + K_ff r(t) + compensator(x_plant)

Actor-critic upgrade (Identity 4 -- DPG connection): the feedback gain is
initialized from the LQR solution :math:`K_{opt} = R^{-1} B^\top P_{opt}` and
improved via the IRL Bellman update. The CLF-CBF safety filter (Identity 3)
wraps the nominal control output. The critic (Identity 1) and costate head
(Identity 2) are attached for value/costate computation and warm-start.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from pits_mras.controllers.reference_models import LinearReferenceModel
from pits_mras.controllers.safety import CLFCBFSafetyFilter
from pits_mras.models.critic import CostateHead, QuadraticCritic
from pits_mras.utils.lyapunov import solve_care


class MRASController(nn.Module):
    """Adaptive controller fusing classical MRAS with the actor-critic upgrade."""

    K_fb: Tensor

    def __init__(
        self,
        reference_model: LinearReferenceModel,
        state_dim: int,
        control_dim: int,
        ref_dim: int,
        plant_dim: int,
        use_safety_filter: bool = True,
    ) -> None:
        super().__init__()
        self.reference_model = reference_model
        self.state_dim = state_dim
        self.control_dim = control_dim

        # Critic and costate head (Identity 1 & 2).
        self.critic = QuadraticCritic(state_dim)
        R_inv = reference_model.R_inv
        B = reference_model.B_m  # [state_dim, control_dim]
        self.costate_head = CostateHead(self.critic, R_inv, B)

        # Classical MRAS feedforward gain (learnable).
        self.K_ff = nn.Parameter(torch.zeros(control_dim, ref_dim))
        # Feedback gain (initialized from LQR optimum, improved via IRL).
        self.register_buffer("K_fb", reference_model.K_opt.clone())

        # Optional nonlinear compensator.
        self.compensator = nn.Sequential(
            nn.Linear(plant_dim, 64),
            nn.Tanh(),
            nn.Linear(64, control_dim),
        )

        # Safety filter (instantiated in setup_safety_filter).
        self.use_safety_filter = use_safety_filter
        self.safety_filter: Optional[CLFCBFSafetyFilter] = None

    def setup_safety_filter(
        self, safety_margin: float = 10.0, decay_rate: float = 1.0
    ) -> None:
        """Instantiate the CBF filter from the critic's current ``P``."""
        P = self.critic.extract_P()
        self.safety_filter = CLFCBFSafetyFilter(
            P=P,
            A_m=self.reference_model.A_m,
            B_ctrl=self.reference_model.B_m,
            safety_margin=safety_margin,
            decay_rate=decay_rate,
        )

    def forward(
        self,
        e: Tensor,
        r: Tensor,
        x_plant: Tensor,
        apply_safety: bool = True,
    ) -> Dict[str, Tensor]:
        r"""Compute the MRAS control law.

        ``u = -K_fb e + K_ff r + compensator(x_plant)`` then (optionally) the
        CBF safety filter. Returns a dict with ``u_nom`` and ``u`` (and, when
        the filter is active, ``h_cbf`` and ``slack``).
        """
        u_fb = -e @ self.K_fb.T  # [batch, control_dim]
        u_ff = r @ self.K_ff.T  # [batch, control_dim]
        u_comp = self.compensator(x_plant)  # [batch, control_dim]
        u_nom = u_fb + u_ff + u_comp

        out: Dict[str, Tensor] = {"u_nom": u_nom}

        if apply_safety and self.use_safety_filter and self.safety_filter is not None:
            u_safe, h_val, slack = self.safety_filter(e, u_nom)
            out["u"] = u_safe
            out["h_cbf"] = h_val
            out["slack"] = slack
        else:
            out["u"] = u_nom
        return out

    def lqr_warm_start(self, Q: Tensor, R: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Warm-start so :math:`\hat P = P_{opt}` (LQR solution).

        Sets ``K_fb = K`` (CARE gain) and aligns the critic's ``P`` with the
        CARE solution via ``critic.set_P``. Returns ``(P, K)`` as float32
        tensors.
        """
        A = self.reference_model.A_m.detach().cpu().numpy()
        B = self.reference_model.B_m.detach().cpu().numpy()
        P, K = solve_care(A, B, Q.detach().cpu().numpy(), R.detach().cpu().numpy())
        P_t = torch.tensor(P, dtype=torch.float32)
        K_t = torch.tensor(K, dtype=torch.float32)
        self.K_fb.copy_(K_t)
        self.critic.set_P(P_t)
        return P_t, K_t
