"""Focused tests for the Phase 5 training pipelines (§8).

Covers the three deliverables:
  * pretrain.py  — curriculum lambda schedules hit exact boundary values,
  * irl_trainer.py — offline batch LS recovers a known P_opt,
  * cotrain.py — one episode runs without NaN and steps the critic.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from pits_mras.config import NetworkConfig, PhysicsConfig, PITSMRASConfig
from pits_mras.controllers.mras import MRASController
from pits_mras.controllers.reference_models import LinearReferenceModel
from pits_mras.models import PITNN, QuadraticCritic
from pits_mras.training.cotrain import cotraining_loop
from pits_mras.training.irl_trainer import train_irl_critic
from pits_mras.training.pretrain import (
    data_weight_schedule,
    pretrain_pitnn,
    temporal_weight_schedule,
)


# --------------------------------------------------------------------------- #
# Curriculum lambda schedules (pretrain.py, §8.1).
# --------------------------------------------------------------------------- #
def test_data_weight_stage1a_constant() -> None:
    """Stage 1A (epoch <= stage1_epochs): lambda_data == 0.1."""
    assert data_weight_schedule(1, 1000, 2000) == 0.1
    assert data_weight_schedule(1000, 1000, 2000) == 0.1


def test_data_weight_stage1b_cosine_endpoints() -> None:
    """Cosine anneal 0.1 -> 1.0 across epochs 1001..3000."""
    val_start = data_weight_schedule(1001, 1000, 2000)
    assert 0.1 < val_start < 0.2
    val_end = data_weight_schedule(3000, 1000, 2000)
    assert math.isclose(val_end, 1.0, abs_tol=1e-6)
    # Midpoint (epoch 2000): 0.1 + 0.9 * 0.5 = 0.55.
    val_mid = data_weight_schedule(2000, 1000, 2000)
    assert math.isclose(val_mid, 0.55, abs_tol=1e-6)


def test_data_weight_stage1c_saturates() -> None:
    """After stage 1B the data weight stays at 1.0."""
    assert math.isclose(data_weight_schedule(4000, 1000, 2000), 1.0, abs_tol=1e-6)


def test_temporal_weight_warmup() -> None:
    """Stage 1C linear warm-up: 0 before epoch 3000, final value at 5000."""
    assert temporal_weight_schedule(3000, 2000, 0.5, stage1_epochs=1000) == 0.0
    assert temporal_weight_schedule(2999, 2000, 0.5, stage1_epochs=1000) == 0.0
    val_mid = temporal_weight_schedule(4000, 2000, 0.5, stage1_epochs=1000)
    assert math.isclose(val_mid, 0.25, abs_tol=1e-6)
    val_end = temporal_weight_schedule(5000, 2000, 0.5, stage1_epochs=1000)
    assert math.isclose(val_end, 0.5, abs_tol=1e-6)


# --------------------------------------------------------------------------- #
# Small fixtures.
# --------------------------------------------------------------------------- #
def _small_cfg() -> PITSMRASConfig:
    cfg = PITSMRASConfig()
    cfg.network = NetworkConfig(
        input_dim=2, hidden_dim=16, output_dim=2, lstm_layers=1,
        attention_heads=2, embedding_dim=8,
    )
    cfg.physics = PhysicsConfig(
        n_generalized_coords=1, hamiltonian_hidden=16, dissipation_hidden=8,
    )
    return cfg


def _small_pitnn(cfg: PITSMRASConfig) -> PITNN:
    return PITNN(cfg.network, cfg.physics)


def _make_ref_model() -> LinearReferenceModel:
    A_m = np.array([[0.0, 1.0], [-1.0, -1.0]])
    B_m = np.array([[0.0], [1.0]])
    C_m = np.eye(2)
    Q = np.eye(2)
    R = np.eye(1)
    return LinearReferenceModel(A_m, B_m, C_m, Q, R)


def _make_controller(ref_model: LinearReferenceModel) -> MRASController:
    return MRASController(
        reference_model=ref_model,
        state_dim=2,
        control_dim=1,
        ref_dim=1,
        plant_dim=2,
        use_safety_filter=True,
    )


# --------------------------------------------------------------------------- #
# pretrain_pitnn end-to-end (small).
# --------------------------------------------------------------------------- #
def test_pretrain_runs_multiple_epochs_finite() -> None:
    cfg = _small_cfg()
    pitnn = _small_pitnn(cfg)
    history = pretrain_pitnn(pitnn, cfg, epochs=3, batch_size=8, seed=1)
    assert len(history["total_loss"]) == 3
    assert all(math.isfinite(v) for v in history["total_loss"])
    assert all(math.isfinite(v) for v in history["physics_loss"])
    assert history["lambda_data"][0] == 0.1  # epoch 1 -> Stage 1A


def test_pretrain_reduces_data_loss() -> None:
    """A short run should reduce the data-fit loss (learning happens)."""
    cfg = _small_cfg()
    pitnn = _small_pitnn(cfg)
    history = pretrain_pitnn(pitnn, cfg, epochs=30, batch_size=16, seed=2, lr=1e-2)
    assert history["data_loss"][-1] < history["data_loss"][0]


# --------------------------------------------------------------------------- #
# irl_trainer convergence on a known synthetic case (§8.3).
# --------------------------------------------------------------------------- #
def test_irl_trainer_recovers_p_opt() -> None:
    """Offline batch LS recovers P_opt on consistent closed-loop data.

    Trajectories come from the optimal closed loop e_dot = (A_m - B_m K_opt) e
    with running cost e^T Q e + u^T R u and u = -K_opt e. For this data
    V(e) = e^T P_opt e satisfies the IRL Bellman identity, so the LS fit must
    recover P_opt to within the 1% tolerance.
    """
    ref_model = _make_ref_model()
    critic = QuadraticCritic(state_dim=2)
    critic.set_P(torch.eye(2) * 5.0)  # perturb away from P_opt

    P_hat, converged, n_iters = train_irl_critic(
        critic, ref_model,
        n_trajectories=64, traj_len=40, window_size=5,
        dt=0.01, max_iters=50, tol=0.01, seed=3,
    )
    P_opt = ref_model.P_opt
    rel_err = torch.linalg.norm(P_hat - P_opt) / torch.linalg.norm(P_opt)
    assert converged, f"did not converge; rel_err={rel_err.item():.4f}"
    assert rel_err.item() < 0.01
    assert n_iters >= 1
    # The critic was actually updated to the fit.
    assert torch.allclose(critic.extract_P(), P_hat, atol=1e-5)


# --------------------------------------------------------------------------- #
# cotrain one episode (§8.2).
# --------------------------------------------------------------------------- #
def test_cotrain_no_nan_and_critic_steps() -> None:
    cfg = _small_cfg()
    pitnn = _small_pitnn(cfg)
    ref_model = _make_ref_model()
    controller = _make_controller(ref_model)
    p_before = controller.critic.W_c.weight.detach().clone()
    metrics = cotraining_loop(
        pitnn, controller, ref_model, cfg,
        n_episodes=2, n_steps=8, batch_size=4, irl_window=3, seed=4,
    )
    assert "irl_loss" in metrics
    assert "total_loss" in metrics
    for series in metrics.values():
        if isinstance(series, list):
            for v in series:
                if isinstance(v, float):
                    assert math.isfinite(v)
    p_after = controller.critic.W_c.weight.detach()
    assert not torch.allclose(p_before, p_after)


def test_cotrain_records_critic_convergence_and_it_decreases() -> None:
    """cotrain reports a critic_convergence series; IRL drives P_hat toward P_opt.

    Starting the critic well away from the CARE solution P_opt, the relative
    Frobenius error ||P_hat - P_opt|| / ||P_opt|| recorded each step must end
    below where it started (the IRL critic update is learning).
    """
    cfg = _small_cfg()
    pitnn = _small_pitnn(cfg)
    ref_model = _make_ref_model()
    controller = _make_controller(ref_model)
    # Perturb the critic away from P_opt so convergence is observable.
    controller.critic.set_P(torch.eye(controller.state_dim) * 5.0)
    metrics = cotraining_loop(
        pitnn, controller, ref_model, cfg,
        n_episodes=3, n_steps=12, batch_size=8, irl_window=3,
        critic_lr=5e-2, seed=7,
    )
    assert "critic_convergence" in metrics
    conv = metrics["critic_convergence"]
    assert len(conv) == 3 * 12
    assert all(math.isfinite(v) for v in conv)
    assert conv[-1] < conv[0]


def test_train_irl_critic_gd_converges_from_perturbation() -> None:
    """Offline gradient IRL fit drives a perturbed critic back to P_opt.

    Unlike the in-loop co-training, this fits the critic on FIXED optimal-closed-
    loop data (decoupled from control stability), so the convex IRL Bellman loss
    converges reliably; the returned per-step relative-error history must end
    well below where it started and close to zero.
    """
    from pits_mras.models.critic import QuadraticCritic
    from pits_mras.training.irl_trainer import train_irl_critic_gd

    ref_model = _make_ref_model()
    critic = QuadraticCritic(state_dim=2)
    critic.set_P(torch.eye(2) * 5.0)  # far from P_opt
    history = train_irl_critic_gd(critic, ref_model, seed=0)
    assert all(math.isfinite(v) for v in history)
    assert history[-1] < history[0]
    assert history[-1] < 0.05  # converges close to the CARE solution


def test_cotrain_positivity_regularizer_repairs_indefinite_critic() -> None:
    """The critic positivity term is actually applied to the critic.

    Regression for the wiring bug where ``1e-3 * positivity_loss`` was added to
    the PITNN objective ``l_total`` but, depending only on the critic's ``W_c``,
    its gradient was never stepped (``optimizer_pitnn`` doesn't own ``W_c`` and
    the IRL block's ``zero_grad`` then wiped it). Here the IRL update is disabled
    (``irl_window`` > ``n_steps`` so the window never fills) and HJB/costate/CBF
    are off, so the positivity regularizer is the ONLY thing that can move the
    critic. A seeded indefinite ``P`` must have its minimum eigenvalue driven
    upward.
    """
    cfg = _small_cfg()
    cfg.losses.lambda_hjb = 0.0
    cfg.safety.enable_cbf = False
    pitnn = _small_pitnn(cfg)
    ref_model = _make_ref_model()
    controller = _make_controller(ref_model)
    # Seed an indefinite P (one negative eigenvalue).
    controller.critic.set_P(torch.tensor([[1.0, 0.0], [0.0, -2.0]]))
    lam_before = torch.linalg.eigvalsh(controller.critic.extract_P()).min().item()
    assert lam_before < 0.0  # indefinite to start

    cotraining_loop(
        pitnn, controller, ref_model, cfg,
        n_episodes=1, n_steps=10, batch_size=4, irl_window=50,
        critic_lr=1e-1, seed=0,
    )
    lam_after = torch.linalg.eigvalsh(controller.critic.extract_P()).min().item()
    assert lam_after > lam_before  # positivity drove the critic toward PD


def test_cotrain_hjb_disabled_path() -> None:
    """lambda_hjb == 0 disables the HJB term but the loop still runs."""
    cfg = _small_cfg()
    cfg.losses.lambda_hjb = 0.0
    pitnn = _small_pitnn(cfg)
    ref_model = _make_ref_model()
    controller = _make_controller(ref_model)
    metrics = cotraining_loop(
        pitnn, controller, ref_model, cfg,
        n_episodes=1, n_steps=8, batch_size=4, irl_window=3, seed=5,
    )
    assert all(math.isfinite(v) for v in metrics["total_loss"])
    assert all(v == 0.0 for v in metrics["hjb_loss"])


def test_cotrain_hjb_applied_to_critic_when_enabled() -> None:
    """With HJB enabled and IRL disabled, the HJB residual actually updates the
    critic (regression for the wiring bug where its gradient was discarded).

    Isolation: irl_window > n_steps so the IRL block never fires; the critic is
    seeded PD but off P_opt, so the positivity step is a no-op while the HJB
    residual is non-zero -> the HJB critic step is the ONLY thing that can move
    W_c.
    """
    cfg = _small_cfg()
    cfg.losses.lambda_hjb = 0.5
    cfg.safety.enable_cbf = False
    pitnn = _small_pitnn(cfg)
    ref_model = _make_ref_model()
    controller = _make_controller(ref_model)
    # PD but not the HJB optimum -> positivity inactive, HJB residual non-zero.
    controller.critic.set_P(torch.eye(controller.state_dim) * 3.0)
    w_before = controller.critic.W_c.weight.detach().clone()
    cotraining_loop(
        pitnn, controller, ref_model, cfg,
        n_episodes=1, n_steps=6, batch_size=4, irl_window=50,
        critic_lr=1e-2, seed=0,
    )
    w_after = controller.critic.W_c.weight.detach()
    assert not torch.allclose(w_before, w_after)  # HJB gradient was applied


def test_cotrain_hjb_default_off_is_irl_only() -> None:
    """The default LossConfig leaves HJB off, so it is never applied."""
    cfg = _small_cfg()  # cfg.losses.lambda_hjb defaults to 0.0
    assert cfg.losses.lambda_hjb == 0.0
    pitnn = _small_pitnn(cfg)
    ref_model = _make_ref_model()
    controller = _make_controller(ref_model)
    metrics = cotraining_loop(
        pitnn, controller, ref_model, cfg,
        n_episodes=1, n_steps=8, batch_size=4, irl_window=3, seed=5,
    )
    assert all(v == 0.0 for v in metrics["hjb_loss"])


def test_cotrain_converges_with_hjb_enabled() -> None:
    """Enabling HJB does not break IRL convergence of the critic toward P_opt."""
    cfg = _small_cfg()
    cfg.losses.lambda_hjb = 0.01
    pitnn = _small_pitnn(cfg)
    ref_model = _make_ref_model()
    controller = _make_controller(ref_model)
    controller.critic.set_P(torch.eye(controller.state_dim) * 5.0)
    metrics = cotraining_loop(
        pitnn, controller, ref_model, cfg,
        n_episodes=3, n_steps=12, batch_size=8, irl_window=3,
        critic_lr=5e-2, seed=7,
    )
    conv = metrics["critic_convergence"]
    assert conv[-1] < conv[0]


# --------------------------------------------------------------------------- #
# irl_trainer non-convergence branch (§8.3).
# --------------------------------------------------------------------------- #
def test_irl_trainer_non_convergence_exhausts_max_iters() -> None:
    """An unreachable tolerance exits at ``max_iters`` with converged=False.

    Exercises the loop's max-iters fall-through (the negative path of the
    ``rel_err < tol`` break) which the convergent test never reaches.
    """
    ref_model = _make_ref_model()
    critic = QuadraticCritic(state_dim=2)
    P_hat, converged, n_iters = train_irl_critic(
        critic, ref_model, tol=1e-12, max_iters=3, seed=0,
    )
    assert converged is False
    assert n_iters == 3
    assert P_hat.shape == (2, 2)
    # Even without converging, the critic is updated to the last LS fit.
    assert torch.allclose(critic.extract_P(), P_hat, atol=1e-5)


# --------------------------------------------------------------------------- #
# pretrain validation-guard and Stage-1C temporal branches (§8.1).
# --------------------------------------------------------------------------- #
def test_pretrain_spike_halves_data_weight() -> None:
    """A physics spike above ``epsilon_tol`` halves ``lambda_data`` (§8.1 guard).

    Forcing ``epsilon_tol`` below any achievable residual fires the safeguard
    on every epoch, so the Stage-1A weight 0.1 is halved to 0.05.
    """
    cfg = _small_cfg()
    pitnn = _small_pitnn(cfg)
    history = pretrain_pitnn(
        pitnn, cfg, epochs=2, batch_size=8, seed=1, epsilon_tol=1e-12,
    )
    assert all(math.isclose(v, 0.05, abs_tol=1e-9) for v in history["lambda_data"])


def test_pretrain_stage1c_activates_temporal_term() -> None:
    """In Stage 1C the temporal loss is weighted (``lambda_temp > 0`` branch).

    With tiny stage boundaries the run crosses into Stage 1C, so the temporal
    warm-up ramps above zero and the ``total += lambda_temp * l_temporal``
    branch executes.
    """
    cfg = _small_cfg()
    cfg.losses.lambda_temporal = 0.5
    cfg.training.stage1_epochs = 1
    cfg.training.stage2_epochs = 1
    pitnn = _small_pitnn(cfg)
    history = pretrain_pitnn(pitnn, cfg, epochs=4, batch_size=8, seed=2)
    # Stage 1A/1B (epochs 1-2) -> 0; Stage 1C (epochs 3-4) -> ramps up.
    assert history["lambda_temp"][0] == 0.0
    assert history["lambda_temp"][-1] > 0.0
    assert all(math.isfinite(v) for v in history["total_loss"])
