"""Co-training pipeline with IRL critic updates (Algorithm 3 extended, §8.2).

This is the closed-loop actor-critic training loop -- "the most critical
training file." The source (§8.2) gives the *additions* to Algorithm 3 but the
base loop body is prose-only (Gap G5); the loop below is constructed here to
wire those additions into a coherent, minimal-but-correct training step against
the REAL Phase 1-4 APIs (the spec's pseudo-code signatures do not match them).

Per step the loop:

1. runs the PITNN forward pass on a synthetic history window to get ``context``,
2. computes the tracking error ``e = x_p - x_m`` and runs the MRAS controller
   to get the safety-filtered control ``u_safe``,
3. accumulates ``(e, u_safe)`` into a rolling IRL window; once the window holds
   ``irl_window + 1`` samples it forms the integral-RL Bellman loss
   (:class:`IRLBellmanLoss`) and takes a policy-evaluation gradient step on the
   critic with a SEPARATE ``critic_optimizer`` (Adam lr=1e-3, grad-clip 1.0),
   then performs the policy-improvement read-out ``K = R^{-1} B^T P_hat``,
4. builds the PITNN objective ``L_total`` = physics + (optional) HJB residual +
   costate consistency + ``1e-3`` * critic positivity + ``0.1`` * CBF
   constraint, and steps the PITNN optimizer (Adam lr=1e-4),
5. advances the synthetic plant + reference model and slides the history window.

Everything runs on tiny synthetic trajectories (no external dataset; Gap G7) so
a single short episode produces finite losses in well under a second.

Design notes (NOT pinned by the source; flagged):

* Function signature, ``n_episodes`` / ``n_steps`` overrides, the synthetic
  plant rollout, the rolling IRL window, and the returned metrics dict are
  designed here.
* The real :class:`IRLBellmanLoss` is window-based (``forward(critic, e, u,
  dt)`` over ``[batch, T+1, dim]``), NOT push/is_ready -- so an explicit rolling
  buffer is maintained instead of the spec's ``irl_accumulator``.
* The running cost ``r(e, u) = e^T Q e + u^T R u`` is computed inline (the spec
  references a non-existent ``irl_loss.running_cost``).
* The policy-improvement gain ``K_new`` is computed for diagnostics only; per
  the source's own annotation the effective feedback gain already lives in the
  costate head (``u_fb = -R^{-1} B^T grad_V``), so no explicit ``K_fb`` is
  mutated.
* The IRL critic step is taken AFTER the PITNN ``L_total`` step so the in-place
  critic-parameter update cannot invalidate the ``L_total`` autograd graph (the
  IRL loss builds its own graph from the rolling window).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Deque

import torch
from torch import Tensor

from pits_mras.losses.hjb import HJBResidualLoss
from pits_mras.losses.irl import IRLBellmanLoss

if TYPE_CHECKING:
    from pits_mras.config import PITSMRASConfig
    from pits_mras.controllers.mras import MRASController
    from pits_mras.controllers.reference_models import LinearReferenceModel
    from pits_mras.models import PITNN

logger = logging.getLogger(__name__)


def _synthetic_plant_step(
    x_p: Tensor, u: Tensor, A_m: Tensor, B_m: Tensor, dt: float
) -> Tensor:
    """Advance a synthetic plant ``x_dot = A_m x + B_m u`` by one Euler step.

    Uses the reference-model matrices as a stand-in plant so the closed loop is
    well-posed without an external dataset. The next state is detached so each
    step's losses are computed on the current detached state.
    """
    x_dot = torch.einsum("ij,bj->bi", A_m, x_p) + torch.einsum("ij,bj->bi", B_m, u)
    return (x_p + dt * x_dot).detach()


def cotraining_loop(
    pitnn: "PITNN",
    controller: "MRASController",
    ref_model: "LinearReferenceModel",
    cfg: "PITSMRASConfig",
    *,
    n_episodes: int | None = None,
    n_steps: int | None = None,
    batch_size: int = 8,
    pitnn_lr: float = 1e-4,
    critic_lr: float = 1e-3,
    irl_window: int = 4,
    history_length: int = 8,
    seed: int | None = None,
) -> dict[str, list[float]]:
    """Closed-loop actor-critic co-training (Algorithm 3 extended, §8.2).

    Args:
        pitnn: dynamics model (updated in place by the PITNN optimizer).
        controller: MRAS controller; its critic is updated by the IRL optimizer.
            A CLF-CBF safety filter is set up from the critic if the controller
            uses one and none is attached yet.
        ref_model: linear reference model and LQR ground truth.
        cfg: full configuration (``cfg.losses`` / ``cfg.safety`` are read).
        n_episodes: number of episodes (defaults to ``cfg.training.n_episodes``).
        n_steps: steps per episode (defaults from ``sim_duration / dt``).
        batch_size: number of parallel synthetic trajectories.
        pitnn_lr: Adam learning rate for the PITNN (source: 1e-4).
        critic_lr: Adam learning rate for the critic (source: 1e-3).
        irl_window: IRL integral-RL window length (in steps).
        history_length: length of the synthetic PITNN history window.
        seed: optional RNG seed.

    Returns:
        Metrics dict mapping names to per-step lists: ``irl_loss``,
        ``hjb_loss``, ``costate_loss``, ``positivity_loss``, ``cbf_loss``,
        ``total_loss``, ``running_cost``.
    """
    train_cfg = cfg.training
    loss_cfg = cfg.losses
    safety_cfg = cfg.safety

    if seed is not None:
        torch.manual_seed(seed)

    n_episodes = train_cfg.n_episodes if n_episodes is None else n_episodes
    if n_steps is None:
        n_steps = max(int(train_cfg.sim_duration / train_cfg.dt), 1)
    dt = train_cfg.dt

    state_dim = controller.state_dim
    control_dim = controller.control_dim
    input_dim = pitnn.input_dim
    output_dim = pitnn.output_dim
    n_q = pitnn.n_q  # PITNN's wired control dimension

    A_m = ref_model.A_m
    B_m = ref_model.B_m
    Q = ref_model.Q
    R = ref_model.R

    # Set up the CBF filter from the critic if requested and not yet attached.
    use_cbf = (
        safety_cfg.enable_cbf
        and controller.use_safety_filter
    )
    if use_cbf and controller.safety_filter is None:
        controller.setup_safety_filter(
            safety_margin=safety_cfg.safety_margin,
            decay_rate=safety_cfg.cbf_decay_rate,
        )

    irl_loss = IRLBellmanLoss(Q, R)
    hjb_loss = HJBResidualLoss(A_m, B_m, Q, R, weight=1.0)

    optimizer_pitnn = torch.optim.Adam(pitnn.parameters(), lr=pitnn_lr)
    critic_optimizer = torch.optim.Adam(controller.critic.parameters(), lr=critic_lr)

    metrics: dict[str, list[float]] = {
        "irl_loss": [],
        "hjb_loss": [],
        "costate_loss": [],
        "positivity_loss": [],
        "cbf_loss": [],
        "total_loss": [],
        "running_cost": [],
    }

    for episode in range(n_episodes):
        # Initialize synthetic plant / reference-model states and history.
        x_p = torch.randn(batch_size, input_dim)
        x_m = torch.zeros(batch_size, input_dim)
        x_hist = torch.zeros(batch_size, history_length, input_dim)
        u_hist = torch.zeros(batch_size, history_length, input_dim)
        e_hist = torch.zeros(batch_size, history_length, output_dim)
        # Rolling IRL window of detached (e, u_safe) samples.
        e_window: Deque[Tensor] = deque(maxlen=irl_window + 1)
        u_window: Deque[Tensor] = deque(maxlen=irl_window + 1)

        for _step in range(n_steps):
            r = torch.randn(batch_size, control_dim)
            u_curr = torch.zeros(batch_size, n_q)

            e_state = x_p - x_m  # [batch, input_dim] full-state error
            e_curr = e_state[:, :output_dim]
            # PITNN forward: its f_hat feeds the physics term below. (The real
            # MRASController.forward takes e/r/x_plant -- NOT the PITNN context
            # the §8.2 pseudo-code references -- so the context is not consumed
            # by the controller in this wiring.)
            pitnn_output = pitnn(x_hist, u_hist, x_p, u_curr, e_curr, e_hist)

            # Controller acts on the reduced tracking error e (state_dim).
            e = e_state[:, :state_dim]
            controller_output = controller(e, r, x_p, apply_safety=use_cbf)
            u_safe = controller_output["u"]

            # Running cost r(e, u) = e^T Q e + u^T R u.
            r_inst = torch.einsum("bi,ij,bj->b", e, Q, e) + torch.einsum(
                "bi,ij,bj->b", u_safe, R, u_safe
            )

            # ── PITNN objective L_total ──
            f_target = torch.einsum("ij,bj->bi", A_m, e) + torch.einsum(
                "ij,bj->bi", B_m, u_safe.detach()
            )
            l_phys = (pitnn_output["f_hat"][:, :state_dim] - f_target).pow(2).mean()
            l_total = loss_cfg.lambda_physics * l_phys

            # ── NEW: HJB residual update ──
            l_hjb_val = 0.0
            if loss_cfg.lambda_hjb > 0:
                hjb_out = hjb_loss(controller.critic, e.detach())
                l_hjb = hjb_out["loss"]
                l_total = l_total + loss_cfg.lambda_hjb * l_hjb
                l_hjb_val = float(l_hjb.detach())

            # ── NEW: Costate consistency loss ──
            lambda_hat, _ = controller.costate_head(e.detach())
            grad_V = controller.critic.gradient(e.detach())
            l_costate = (lambda_hat - grad_V).pow(2).mean()
            l_total = l_total + loss_cfg.lambda_costate * l_costate

            # ── NEW: Critic positivity regularization ──
            l_pos = controller.critic.positivity_loss()
            l_total = l_total + 1e-3 * l_pos

            # ── CBF constraint loss ──
            l_cbf_val = 0.0
            if use_cbf and controller.safety_filter is not None:
                l_cbf = controller.safety_filter.cbf_constraint_loss(
                    e.detach(), u_safe.detach()
                )
                l_total = l_total + 0.1 * l_cbf
                l_cbf_val = float(l_cbf.detach())

            optimizer_pitnn.zero_grad()
            critic_optimizer.zero_grad()
            l_total.backward()
            optimizer_pitnn.step()

            # ── NEW: IRL critic update (Identity 1 — Vrabie & Lewis 2009) ──
            e_window.append(e.detach())
            u_window.append(u_safe.detach())
            l_irl_val = 0.0
            if len(e_window) == irl_window + 1:
                e_win = torch.stack(list(e_window), dim=1)  # [batch, W+1, n]
                u_win = torch.stack(list(u_window), dim=1)  # [batch, W+1, m]
                irl_out = irl_loss(controller.critic, e_win, u_win, dt)
                l_irl = irl_out["loss"]
                critic_optimizer.zero_grad()
                l_irl.backward()
                torch.nn.utils.clip_grad_norm_(
                    controller.critic.parameters(), max_norm=1.0
                )
                critic_optimizer.step()
                l_irl_val = float(l_irl.detach())
                # Policy-improvement read-out: K <- R^{-1} B^T P_hat (diagnostic).
                with torch.no_grad():
                    P_hat = controller.critic.extract_P()
                    _ = ref_model.R_inv @ ref_model.B_m.T @ P_hat

            metrics["irl_loss"].append(l_irl_val)
            metrics["hjb_loss"].append(l_hjb_val)
            metrics["costate_loss"].append(float(l_costate.detach()))
            metrics["positivity_loss"].append(float(l_pos.detach()))
            metrics["cbf_loss"].append(l_cbf_val)
            metrics["total_loss"].append(float(l_total.detach()))
            metrics["running_cost"].append(float(r_inst.detach().mean()))

            # ── Advance synthetic plant + reference model, slide history ──
            u_full = torch.zeros(batch_size, input_dim)
            u_full[:, :control_dim] = u_safe.detach()
            x_hist = torch.cat([x_hist[:, 1:, :], x_p.unsqueeze(1)], dim=1).detach()
            u_hist = torch.cat(
                [u_hist[:, 1:, :], u_full.unsqueeze(1)], dim=1
            ).detach()
            e_hist = torch.cat(
                [e_hist[:, 1:, :], e_curr.detach().unsqueeze(1)], dim=1
            ).detach()
            x_p = _synthetic_plant_step(x_p, u_full, A_m, B_m, dt)
            x_m = ref_model.step(x_m, r, dt).detach()

        if episode % max(train_cfg.log_every, 1) == 0:
            logger.info(
                "cotrain episode %d/%d  total=%.4g irl=%.4g",
                episode,
                n_episodes,
                metrics["total_loss"][-1],
                metrics["irl_loss"][-1],
            )

    return metrics
