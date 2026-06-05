"""Integration tests: PCML wired into config / losses / PITNN (PCML Addendum §2.4, §3, §4).

These verify the opt-in integration surface -- a `PCMLConfig` on the master
config, a `pcml` component in `TotalLoss`, an optional Lagrangian head on the
PITNN, and a full PITNN -> PCML hard-projection forward pass.
"""

import math

import numpy as np
import torch

from pits_mras.config import LossConfig, NetworkConfig, PCMLConfig, PhysicsConfig, PITSMRASConfig
from pits_mras.constraints import HeatConductionDAE
from pits_mras.controllers.mras import MRASController
from pits_mras.controllers.reference_models import LinearReferenceModel
from pits_mras.inference.realtime import RealtimeInferenceEngine
from pits_mras.losses import TotalLoss
from pits_mras.models.lagrangian_head import LagrangianMultiplierHead
from pits_mras.models.pcml import PCMLModule
from pits_mras.models.pitnn import PITNN
from pits_mras.training.cotrain import cotraining_loop


# --------------------------------------------------------------------------- #
# Config.
# --------------------------------------------------------------------------- #
def test_master_config_has_pcml() -> None:
    cfg = PITSMRASConfig()
    assert isinstance(cfg.pcml, PCMLConfig)
    # A couple of DAE-HardNet defaults.
    assert cfg.pcml.eta == 0.01
    assert cfg.pcml.omega == 1.0
    assert cfg.pcml.delta == 0.01


def test_loss_config_has_lambda_pcml() -> None:
    assert hasattr(LossConfig(), "lambda_pcml")


# --------------------------------------------------------------------------- #
# TotalLoss with a pcml component.
# --------------------------------------------------------------------------- #
def test_total_loss_includes_pcml_component() -> None:
    cfg = LossConfig()
    cfg.lambda_pcml = 2.0
    total = TotalLoss(cfg)
    out = total({"data": torch.tensor(1.0), "pcml": torch.tensor(3.0)})
    assert "loss/pcml" in out
    assert torch.allclose(out["loss/pcml"], torch.tensor(6.0))  # 2.0 * 3.0
    # total = data (1.0 * default lambda_data=1.0) + pcml (6.0)
    assert torch.allclose(out["loss"], torch.tensor(7.0))


# --------------------------------------------------------------------------- #
# PITNN optional Lagrangian head.
# --------------------------------------------------------------------------- #
def _small_pitnn(lagrangian_head=None) -> PITNN:
    net = NetworkConfig(
        input_dim=6,
        hidden_dim=16,
        output_dim=4,
        lstm_layers=1,
        attention_heads=2,
        embedding_dim=8,
    )
    phys = PhysicsConfig(n_generalized_coords=2)
    return PITNN(net, phys, lagrangian_head=lagrangian_head)


def _pitnn_inputs(batch=3, T=5, input_dim=6, out_dim=4, n_ctrl=2):
    return dict(
        x_hist=torch.randn(batch, T, input_dim),
        u_hist=torch.randn(batch, T, input_dim),
        x_p_curr=torch.randn(batch, input_dim),
        u_curr=torch.randn(batch, n_ctrl),
        e_curr=torch.randn(batch, out_dim),
        e_hist=torch.randn(batch, T, out_dim),
    )


def test_pitnn_without_lagrangian_head_has_no_lam_hat() -> None:
    torch.manual_seed(0)
    pitnn = _small_pitnn()
    out = pitnn(**_pitnn_inputs())
    assert "lam_hat" not in out


def test_pitnn_with_lagrangian_head_emits_lam_hat() -> None:
    torch.manual_seed(0)
    head = LagrangianMultiplierHead(context_dim=16, n_lambda_eq=1, n_lambda_ineq=2)
    pitnn = _small_pitnn(lagrangian_head=head)
    out = pitnn(**_pitnn_inputs())
    assert "lam_hat" in out
    assert out["lam_hat"].shape == (3, 3)  # 1 eq + 2 ineq


# --------------------------------------------------------------------------- #
# Full PITNN -> PCML hard projection forward (integration smoke).
# --------------------------------------------------------------------------- #
def test_pcml_hard_forward_on_heat_constraints() -> None:
    torch.manual_seed(0)
    dae = HeatConductionDAE(alpha=1.0, T_min=-100.0, T_max=100.0)
    backbone = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    n_lambda = dae.spec.n_differential + dae.spec.n_inequality
    mod = PCMLModule(
        constraints=dae,
        backbone=backbone,
        input_dim=2,
        n_output=1,
        n_deriv=4,
        n_lambda=n_lambda,
        eta=0.5,
    )
    mod.update_activation(0.1)  # below eta -> hard mode
    assert mod.mode == "hard"
    x = torch.zeros(4, 1)
    t = torch.zeros(4, 1)
    y_hat = torch.randn(4, 1) * 3.0
    d_hat = torch.randn(4, 4)
    lam_hat = torch.zeros(4, n_lambda)
    y_pcml, loss, info = mod(x, t, y_hat, d_hat, lam_hat, y_true=torch.randn(4, 1))
    assert y_pcml.shape == (4, 1)
    assert info["mode"] == "hard"
    # The projected output satisfies the heat equation to high precision.
    assert info["violation"].item() < 1e-4


# --------------------------------------------------------------------------- #
# Opt-in loop hooks: cotrain mode-switch + realtime projection bypass.
# --------------------------------------------------------------------------- #
def _small_cfg() -> PITSMRASConfig:
    cfg = PITSMRASConfig()
    cfg.network = NetworkConfig(
        input_dim=2,
        hidden_dim=16,
        output_dim=2,
        lstm_layers=1,
        attention_heads=2,
        embedding_dim=8,
    )
    cfg.physics = PhysicsConfig(
        n_generalized_coords=1,
        hamiltonian_hidden=16,
        dissipation_hidden=8,
    )
    return cfg


def _ref_model() -> LinearReferenceModel:
    A_m = np.array([[0.0, 1.0], [-1.0, -1.0]])
    B_m = np.array([[0.0], [1.0]])
    return LinearReferenceModel(A_m, B_m, np.eye(2), np.eye(2), np.eye(1))


def _heat_pcml(eta: float) -> PCMLModule:
    dae = HeatConductionDAE(alpha=1.0, T_min=-1e3, T_max=1e3)
    backbone = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    n_lambda = dae.spec.n_differential + dae.spec.n_inequality
    return PCMLModule(
        constraints=dae,
        backbone=backbone,
        input_dim=2,
        n_output=1,
        n_deriv=4,
        n_lambda=n_lambda,
        eta=eta,
        max_newton_iter=8,
    )


def test_cotrain_with_pcml_module_runs() -> None:
    cfg = _small_cfg()
    rm = _ref_model()
    pitnn = PITNN(cfg.network, cfg.physics)
    ctrl = MRASController(rm, state_dim=2, control_dim=1, ref_dim=1, plant_dim=2)
    pcml = _heat_pcml(eta=1e-12)  # stays in soft mode for the loop
    metrics = cotraining_loop(
        pitnn,
        ctrl,
        rm,
        cfg,
        n_episodes=1,
        n_steps=3,
        batch_size=2,
        seed=0,
        pcml_module=pcml,
    )
    assert "pcml_loss" in metrics
    assert len(metrics["pcml_loss"]) == 3
    assert all(math.isfinite(v) for v in metrics["pcml_loss"])


def test_realtime_with_pcml_projection_emits_violation() -> None:
    cfg = _small_cfg()
    rm = _ref_model()
    pitnn = PITNN(cfg.network, cfg.physics)
    ctrl = MRASController(rm, state_dim=2, control_dim=1, ref_dim=1, plant_dim=2)
    ctrl.setup_safety_filter()
    pcml = _heat_pcml(eta=1.0)
    pcml.update_activation(0.0)  # force hard mode -> projection path
    engine = RealtimeInferenceEngine(pitnn, ctrl, rm, horizon=5, pcml_module=pcml)
    out = engine.step(torch.zeros(2), torch.ones(1), dt=0.01)
    assert "pcml_violation" in out
    assert math.isfinite(out["pcml_violation"])
