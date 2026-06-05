r"""Actor-critic MRAS controller (IP §7.3). Identities 1, 2, 3, 4.

Owning phase: Phase 4 (Controllers).

Control law::

    u(t) = u_fb(e) + K_ff r(t) + compensator(x_plant)

where the feedback ``u_fb`` is the **costate-head optimal control**
:math:`u_{fb} = -\tfrac12 R^{-1}B^\top\nabla\hat V = -R^{-1}B^\top\hat P e`
(Identity 2), so as the IRL critic learns :math:`\hat P`, the actuator follows
it (Identity 4 actor-critic fusion). The critic is warm-started to the LQR/CARE
solution :math:`P_{opt}` at construction, so ``u_fb`` equals the LQR gain
:math:`-K_{opt} e` at initialization and adapts thereafter. The CLF-CBF safety
filter (Identity 3) wraps the nominal control. ``K_fb`` is retained as the
LQR warm-start gain for reference and :meth:`lqr_warm_start`, but the live
feedback comes from the costate head, not that frozen buffer.

The DPG actor-update half of Identity 4 (regressor :math:`\phi_c=[e,r,x_p]` and
the deterministic-policy-gradient step) is provided by :meth:`mras_regressor`
and :meth:`dpg_actor_step`.
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

        # Critic and costate head (Identity 1 & 2). Warm-start the critic to the
        # LQR/CARE solution P_opt so the costate-head feedback equals -K_opt e at
        # initialization and then adapts as the IRL critic learns (Identity 4).
        self.critic = QuadraticCritic(state_dim)
        self.critic.set_P(reference_model.P_opt)
        R_inv = reference_model.R_inv
        B = reference_model.B_m  # [state_dim, control_dim]
        self.costate_head = CostateHead(self.critic, R_inv, B)

        # Classical MRAS feedforward gain (learnable).
        self.K_ff = nn.Parameter(torch.zeros(control_dim, ref_dim))
        # LQR warm-start gain, retained for reference / lqr_warm_start. The live
        # feedback now comes from the costate head, not this frozen buffer.
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

    def setup_safety_filter(self, safety_margin: float = 10.0, decay_rate: float = 1.0) -> None:
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

        ``u = u_fb(e) + K_ff r + compensator(x_plant)`` then (optionally) the
        CBF safety filter, where ``u_fb`` is the costate-head optimal control
        :math:`-\tfrac12 R^{-1}B^\top\nabla\hat V` (Identity 2). Returns a dict
        with ``u_nom``, ``u``, ``lambda_hat`` (the costate :math:`\nabla\hat V`)
        and ``v_hat`` (the value :math:`\hat V(e)`); when the filter is active,
        also ``h_cbf`` and ``slack``.

        Note: the costate uses ``torch.autograd.grad`` internally, so callers
        under ``torch.no_grad()`` must wrap this in ``torch.enable_grad()`` (the
        real-time engine does) and detach the result.
        """
        lambda_hat, u_fb = self.costate_head(e)  # u_fb = -½R⁻¹Bᵀ∇V̂ = -R⁻¹BᵀP̂e
        u_ff = r @ self.K_ff.T  # [batch, control_dim]
        u_comp = self.compensator(x_plant)  # [batch, control_dim]
        u_nom = u_fb + u_ff + u_comp

        out: Dict[str, Tensor] = {
            "u_nom": u_nom,
            "lambda_hat": lambda_hat,
            "v_hat": self.critic(e),
        }

        if apply_safety and self.use_safety_filter and self.safety_filter is not None:
            u_safe, h_val, slack = self.safety_filter(e, u_nom)
            out["u"] = u_safe
            out["h_cbf"] = h_val
            out["slack"] = slack
        else:
            out["u"] = u_nom
        return out

    def lqr_warm_start(self, Q: Tensor, R: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Re-warm-start the critic to the LQR/CARE solution for a *given* cost.

        ``__init__`` already warm-starts the critic to ``reference_model.P_opt``
        — the CARE solution for the reference model's *own* ``Q``/``R``. This
        method is the public way to re-warm-start to a **different** cost: it
        re-solves CARE for the caller-supplied ``Q``/``R``, sets ``K_fb = K``
        (the new CARE gain) and aligns the critic's ``P`` via ``critic.set_P``.
        With the construction-time ``Q``/``R`` it reproduces the initial
        warm-start; with a different cost it does not (it is *not* redundant
        with the constructor). Returns ``(P, K)`` as float32 tensors.
        """
        A = self.reference_model.A_m.detach().cpu().numpy()
        B = self.reference_model.B_m.detach().cpu().numpy()
        P, K = solve_care(A, B, Q.detach().cpu().numpy(), R.detach().cpu().numpy())
        P_t = torch.tensor(P, dtype=torch.float32)
        K_t = torch.tensor(K, dtype=torch.float32)
        self.K_fb.copy_(K_t)
        self.critic.set_P(P_t)
        return P_t, K_t

    def mras_regressor(self, e: Tensor, r: Tensor, x_plant: Tensor) -> Tensor:
        r"""Classical MRAS regressor :math:`\phi_c = [e^\top, r^\top, x_p^\top]^\top`.

        Used by both the classical and DPG-style adaptation laws (IP §7.3).
        Returns ``[batch, state_dim + ref_dim + plant_dim]``.
        """
        return torch.cat([e, r, x_plant], dim=-1)

    def dpg_action_value_gradient(self, e: Tensor, u: Tensor) -> Tensor:
        r"""Action-value gradient :math:`\nabla_a \hat Q(e, u) = R u + B^\top \hat P e`.

        This is the (half) gradient of the LQR action-value function w.r.t. the
        action. It vanishes at the optimal control :math:`u^* = -R^{-1}B^\top\hat P e`
        (the costate-head output), confirming backward compatibility with the
        LQR limit -- IP §3.6's "``∇_a Q̂`` reduces to ``Pe`` (up to factor 2)".
        ``\hat P`` is read from the live critic, so the gradient tracks the
        learned value function. Shapes: ``e`` ``[batch, state_dim]``, ``u``
        ``[batch, control_dim]`` -> ``[batch, control_dim]``.
        """
        P_hat = self.critic.extract_P()  # [state_dim, state_dim]
        R = self.reference_model.R  # [control_dim, control_dim]
        B = self.reference_model.B_m  # [state_dim, control_dim]
        return u @ R + (e @ P_hat) @ B  # R u + B^T P_hat e (row form)

    def dpg_actor_step(self, e: Tensor, r: Tensor, x_plant: Tensor, gamma_c: float = 0.1) -> Tensor:
        r"""Deterministic-policy-gradient actor update (Identity 4, IP §3.6).

        Adds the DPG term :math:`\Gamma_c\,\nabla_\theta u \cdot \nabla_a\hat Q`
        to the gradients of the actor parameters (``K_ff`` and the
        ``compensator``), to be applied by the caller's optimizer *after* the
        standard gradient step. The costate-head feedback is the critic's
        responsibility (trained by IRL), so it is detached here -- the DPG step
        improves only the actor params, mirroring standard actor-critic.

        Realized via the surrogate ``gamma_c * (u * stopgrad(∇_a Q̂)).sum()``
        whose parameter-gradient is exactly the deterministic policy gradient.
        Returns the scalar surrogate (already back-propagated).
        """
        _, u_fb = self.costate_head(e)
        u_actor = u_fb.detach() + r @ self.K_ff.T + self.compensator(x_plant)
        q_grad_a = self.dpg_action_value_gradient(e, u_actor).detach()
        surrogate = gamma_c * (u_actor * q_grad_a).sum(dim=-1).mean()
        surrogate.backward()
        return surrogate.detach()
