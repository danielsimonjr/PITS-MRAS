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
4. builds the PITNN objective ``L_total`` = physics + (optional) PCML +
   ``cfg.losses.lambda_cbf`` * CBF constraint, and steps the PITNN optimizer
   (Adam lr=1e-4); the
   critic-only regularizers are applied SEPARATELY through the *critic* optimizer
   — an opt-in HJB residual (``lambda_hjb`` > 0) and positive-definiteness
   (``_POSITIVITY_WEIGHT`` * ``relu(-λ_min(P))``),
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
    from pits_mras.config import LossConfig, PITSMRASConfig
    from pits_mras.controllers.mras import MRASController
    from pits_mras.controllers.reference_models import LinearReferenceModel
    from pits_mras.controllers.safety import CLFCBFSafetyFilter
    from pits_mras.models import PITNN
    from pits_mras.models.critic import QuadraticCritic
    from pits_mras.models.pcml import PCMLModule

logger = logging.getLogger(__name__)

# Weight on the critic positive-definiteness regularizer (``relu(-λ_min(P))``).
# Applied through the critic optimizer; see the co-training loop. Module-internal.
_POSITIVITY_WEIGHT = 1e-3


def _synthetic_plant_step(x_p: Tensor, u: Tensor, A_m: Tensor, B_m: Tensor, dt: float) -> Tensor:
    """Advance a synthetic plant ``x_dot = A_m x + B_m u`` by one Euler step.

    Uses the reference-model matrices as a stand-in plant so the closed loop is
    well-posed without an external dataset. The next state is detached so each
    step's losses are computed on the current detached state.
    """
    x_dot = torch.einsum("ij,bj->bi", A_m, x_p) + torch.einsum("ij,bj->bi", B_m, u)
    return (x_p + dt * x_dot).detach()


def _pitnn_objective_step(
    pitnn_output: dict[str, Tensor],
    e: Tensor,
    u_safe: Tensor,
    f_target: Tensor,
    state_dim: int,
    loss_cfg: "LossConfig",
    optimizer_pitnn: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    *,
    pcml_module: "PCMLModule | None",
    use_cbf: bool,
    safety_filter: "CLFCBFSafetyFilter | None",
    batch_size: int,
    x_p: Tensor,
) -> tuple[float, float, float]:
    """Build the PITNN objective ``L_total`` and take ONE PITNN optimizer step.

    ``L_total`` = ``lambda_physics`` * physics + (optional) ``lambda_pcml`` * PCML
    + (optional) ``lambda_cbf`` * CBF. These are exactly the components that ride
    the *PITNN* optimizer; the critic-only regularizers (HJB / positivity / IRL)
    are stepped separately by the caller through ``critic_optimizer``.

    The PITNN gradients are zeroed, ``L_total.backward()`` is run, and the PITNN
    optimizer is stepped. ``critic_optimizer.zero_grad()`` is also called first so
    the critic graph stays clean (``L_total`` is a pure PITNN objective; any
    gradient it deposits on the critic's ``W_c`` is discarded here).

    Returns ``(l_total_val, l_pcml_val, l_cbf_val)`` as plain floats for logging.
    """
    # ── physics term ──
    l_phys = (pitnn_output["f_hat"][:, :state_dim] - f_target).pow(2).mean()
    l_total = loss_cfg.lambda_physics * l_phys

    # ── PCML constraint loss (opt-in; §3.1 dynamic activation) ──
    l_pcml_val = 0.0
    if pcml_module is not None:
        n_out = pcml_module.projection.n_y
        n_der = pcml_module.n_deriv
        zeros_xt = x_p.new_zeros(batch_size, 1)
        y_hat_p = pitnn_output["f_hat"][:, :n_out]
        d_hat_p = pitnn_output["f_hat"].new_zeros(batch_size, n_der)
        lam_hat_p = pitnn_output.get(
            "lam_hat",
            pitnn_output["f_hat"].new_zeros(
                batch_size, pcml_module.projection.n_c + pcml_module.projection.n_g
            ),
        )
        pcml_module.update_activation(float(l_phys.detach()))
        _, l_pcml, _ = pcml_module(
            zeros_xt,
            zeros_xt,
            y_hat_p,
            d_hat_p,
            lam_hat_p,
            y_true=f_target[:, :n_out],
        )
        l_total = l_total + loss_cfg.lambda_pcml * l_pcml
        l_pcml_val = float(l_pcml.detach())

    # ── CBF constraint loss (opt-in via the safety filter) ──
    l_cbf_val = 0.0
    if use_cbf and safety_filter is not None:
        l_cbf = safety_filter.cbf_constraint_loss(e.detach(), u_safe.detach())
        l_total = l_total + loss_cfg.lambda_cbf * l_cbf
        l_cbf_val = float(l_cbf.detach())

    optimizer_pitnn.zero_grad()
    critic_optimizer.zero_grad()  # l_total is a pure PITNN objective
    l_total.backward()
    optimizer_pitnn.step()

    return float(l_total.detach()), l_pcml_val, l_cbf_val


def _hjb_critic_step(
    critic: torch.nn.Module,
    e: Tensor,
    hjb_loss: HJBResidualLoss,
    lambda_hjb: float,
    critic_optimizer: torch.optim.Optimizer,
) -> float:
    """Opt-in HJB residual regularizer on the CRITIC (applies iff ``lambda_hjb > 0``).

    HJB depends only on the critic's ``W_c``, so it must ride with
    ``critic_optimizer`` — in ``L_total`` its gradient went to ``W_c`` unstepped
    and was wiped. Unlike the positivity step (guarded because ``positivity_loss``
    is EXACTLY 0 in the healthy regime, so an unguarded step would be a pure no-op
    that still advances Adam's counter), the HJB residual is a genuine signal the
    caller opted into: it applies every step by design. Sharing
    ``critic_optimizer``'s Adam state with the IRL step is the accepted cost of
    co-training the critic with IRL + HJB through one optimizer.

    Returns the HJB residual loss as a float (0.0 when disabled).
    """
    if lambda_hjb <= 0.0:
        return 0.0
    critic_optimizer.zero_grad()
    hjb_out = hjb_loss(critic, e.detach())
    (lambda_hjb * hjb_out["loss"]).backward()
    critic_optimizer.step()
    return float(hjb_out["loss"].detach())


def _positivity_critic_step(
    critic: "QuadraticCritic",
    critic_optimizer: torch.optim.Optimizer,
) -> float:
    """Guarded critic positive-definiteness regularization (applied via the CRITIC
    optimizer).

    It depends only on the critic's ``W_c``, so it must ride with
    ``critic_optimizer`` — adding it to ``L_total`` (PITNN objective) left its
    gradient on ``W_c`` unstepped and then wiped by the IRL block's
    ``zero_grad``. ``positivity_loss`` is exactly 0 while P is PD (the healthy
    regime), so the step is GUARDED on a strictly positive loss: this both skips
    a no-op update and avoids advancing the shared critic-optimizer's Adam step
    count (which would bias the IRL update's bias-correction schedule). When P
    drifts indefinite it pulls the minimum eigenvalue back up.

    Returns the positivity loss value as a float (the metric recorded each step).
    """
    l_pos = critic.positivity_loss()
    l_pos_val = float(l_pos.detach())
    if l_pos_val > 0.0:
        critic_optimizer.zero_grad()
        (_POSITIVITY_WEIGHT * l_pos).backward()
        critic_optimizer.step()
    return l_pos_val


def _irl_critic_step(
    controller: "MRASController",
    ref_model: "LinearReferenceModel",
    e_window: "Deque[Tensor]",
    u_window: "Deque[Tensor]",
    e: Tensor,
    u_safe: Tensor,
    irl_window: int,
    irl_loss: IRLBellmanLoss,
    dt: float,
    critic_optimizer: torch.optim.Optimizer,
) -> float:
    """IRL critic update (Identity 1 — Vrabie & Lewis 2009); window-gated.

    Appends the detached ``(e, u_safe)`` sample to the rolling window; once it
    holds ``irl_window + 1`` samples it forms the integral-RL Bellman loss and
    takes one policy-evaluation gradient step on the critic (grad-clip 1.0) via
    ``critic_optimizer``, then runs the diagnostic policy-improvement read-out
    ``K <- R^{-1} B^T P_hat``.

    Returns the IRL Bellman loss as a float (0.0 until the window fills).
    """
    e_window.append(e.detach())
    u_window.append(u_safe.detach())
    if len(e_window) != irl_window + 1:
        return 0.0
    e_win = torch.stack(list(e_window), dim=1)  # [batch, W+1, n]
    u_win = torch.stack(list(u_window), dim=1)  # [batch, W+1, m]
    irl_out = irl_loss(controller.critic, e_win, u_win, dt)
    l_irl = irl_out["loss"]
    critic_optimizer.zero_grad()
    l_irl.backward()
    torch.nn.utils.clip_grad_norm_(controller.critic.parameters(), max_norm=1.0)
    critic_optimizer.step()
    # Policy-improvement read-out: K <- R^{-1} B^T P_hat (diagnostic).
    with torch.no_grad():
        P_hat = controller.critic.extract_P()
        _ = ref_model.R_inv @ ref_model.B_m.T @ P_hat
    return float(l_irl.detach())


def _advance_plant(
    x_p: Tensor,
    x_m: Tensor,
    x_hist: Tensor,
    u_hist: Tensor,
    e_hist: Tensor,
    e_curr: Tensor,
    u_safe: Tensor,
    r: Tensor,
    ref_model: "LinearReferenceModel",
    A_m: Tensor,
    B_m: Tensor,
    dt: float,
    control_dim: int,
    input_dim: int,
    batch_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Advance the synthetic plant + reference model and slide the history window.

    Returns the updated ``(x_hist, u_hist, e_hist, x_p, x_m)`` tuple. All returned
    tensors are detached (each step's losses are computed on detached state).
    """
    u_full = torch.zeros(batch_size, input_dim)
    u_full[:, :control_dim] = u_safe.detach()
    x_hist = torch.cat([x_hist[:, 1:, :], x_p.unsqueeze(1)], dim=1).detach()
    u_hist = torch.cat([u_hist[:, 1:, :], u_full.unsqueeze(1)], dim=1).detach()
    e_hist = torch.cat([e_hist[:, 1:, :], e_curr.detach().unsqueeze(1)], dim=1).detach()
    x_p = _synthetic_plant_step(x_p, u_full, A_m, B_m, dt)
    x_m = ref_model.step(x_m, r, dt).detach()
    return x_hist, u_hist, e_hist, x_p, x_m


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
    pcml_module: "PCMLModule | None" = None,
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
        pcml_module: optional :class:`~pits_mras.models.pcml.PCMLModule`. When
            supplied, its constraint loss (soft, escalating to the hard KKT
            projection once the data-fit loss drops below ``eta``) is added to
            the PITNN objective weighted by ``cfg.losses.lambda_pcml``, and a
            ``pcml_loss`` metric is recorded. Default ``None`` leaves the v0.2.0
            loop unchanged. NOTE: the synthetic plant has no spatial/temporal
            coordinates, so the constraint inputs ``x``/``t`` and the derivative
            variables ``d`` are passed as zeros -- the constraint residual is
            evaluated on the predicted dynamics ``f_hat`` (and the optional
            ``lam_hat`` warm start), per the §3.1 mode-switch wiring.

    Returns:
        Metrics dict mapping names to per-step lists: ``irl_loss``,
        ``hjb_loss``, ``positivity_loss``, ``cbf_loss``,
        ``total_loss``, ``running_cost``, ``critic_convergence`` (and ``pcml_loss``
        when a ``pcml_module`` is supplied).
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
    # CARE reference for the critic-convergence metric.
    p_opt = ref_model.P_opt
    p_opt_norm = torch.linalg.norm(p_opt)

    # Set up the CBF filter from the critic if requested and not yet attached.
    use_cbf = safety_cfg.enable_cbf and controller.use_safety_filter
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
        "positivity_loss": [],
        "cbf_loss": [],
        "total_loss": [],
        "running_cost": [],
        "critic_convergence": [],
    }
    if pcml_module is not None:
        metrics["pcml_loss"] = []

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

            # PITNN-objective target: f_target = A_m e + B_m u_safe (control detached).
            f_target = torch.einsum("ij,bj->bi", A_m, e) + torch.einsum(
                "ij,bj->bi", B_m, u_safe.detach()
            )

            # ── Step 1: PITNN objective (physics + pcml + cbf) -> PITNN optimizer.
            l_total_val, l_pcml_val, l_cbf_val = _pitnn_objective_step(
                pitnn_output,
                e,
                u_safe,
                f_target,
                state_dim,
                loss_cfg,
                optimizer_pitnn,
                critic_optimizer,
                pcml_module=pcml_module,
                use_cbf=use_cbf,
                safety_filter=controller.safety_filter,
                batch_size=batch_size,
                x_p=x_p,
            )

            # ── Step 2: HJB residual regularizer -> critic optimizer (opt-in).
            l_hjb_val = _hjb_critic_step(
                controller.critic, e, hjb_loss, loss_cfg.lambda_hjb, critic_optimizer
            )

            # ── Step 3: critic positivity regularizer -> critic optimizer (guarded).
            l_pos_val = _positivity_critic_step(controller.critic, critic_optimizer)

            # ── Step 4: IRL critic update -> critic optimizer (window-gated).
            l_irl_val = _irl_critic_step(
                controller,
                ref_model,
                e_window,
                u_window,
                e,
                u_safe,
                irl_window,
                irl_loss,
                dt,
                critic_optimizer,
            )

            # ── Step 5: record per-step metrics.
            metrics["irl_loss"].append(l_irl_val)
            metrics["hjb_loss"].append(l_hjb_val)
            metrics["positivity_loss"].append(l_pos_val)
            metrics["cbf_loss"].append(l_cbf_val)
            metrics["total_loss"].append(l_total_val)
            metrics["running_cost"].append(float(r_inst.detach().mean()))
            # Critic convergence to the CARE solution (reflects the IRL update).
            with torch.no_grad():
                p_hat = controller.critic.extract_P()
                diff = torch.linalg.norm(p_hat - p_opt)
                conv = diff / p_opt_norm if float(p_opt_norm) > 0.0 else diff
            metrics["critic_convergence"].append(float(conv))
            if pcml_module is not None:
                metrics["pcml_loss"].append(l_pcml_val)

            # ── Step 6: advance synthetic plant + reference model, slide history.
            x_hist, u_hist, e_hist, x_p, x_m = _advance_plant(
                x_p,
                x_m,
                x_hist,
                u_hist,
                e_hist,
                e_curr,
                u_safe,
                r,
                ref_model,
                A_m,
                B_m,
                dt,
                control_dim,
                input_dim,
                batch_size,
            )

        if episode % max(train_cfg.log_every, 1) == 0:
            logger.info(
                "cotrain episode %d/%d  total=%.4g irl=%.4g",
                episode,
                n_episodes,
                metrics["total_loss"][-1],
                metrics["irl_loss"][-1],
            )

    return metrics
