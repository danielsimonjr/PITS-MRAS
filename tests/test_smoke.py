"""Smoke test (IP §11.6): end-to-end, no crash / no NaN.

Targets the full stack (``pitnn`` -> controller -> inference) plus the example
scripts. Owning phase: Phase 5 / Phase 6 per ROADMAP.md (authored alongside its
target phase; §11 catalogs it under "Phase 8").

This file provides:
  * ``test_package_imports`` / ``test_example_script_imports`` -- ACTIVE Phase-0
    scaffold gates proving the package and the three ``examples/`` scripts import
    without error (complementing the exhaustive per-module check in
    ``test_imports.py``).
  * ``test_pretrain_one_epoch`` / ``test_cotrain_one_episode`` -- the Phase-5
    acceptance gate (un-skipped): one tiny synthetic step produces finite
    (non-NaN) losses.
  * ``test_full_forward_pass_no_crash`` -- still skipped until Phase 6.
"""

import importlib
import importlib.util
import math
import pathlib

import numpy as np
import pytest
import torch

from pits_mras.config import NetworkConfig, PhysicsConfig, PITSMRASConfig
from pits_mras.controllers.mras import MRASController
from pits_mras.controllers.reference_models import LinearReferenceModel
from pits_mras.models import PITNN

_EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent.parent / "examples"
_EXAMPLE_SCRIPTS = ["robotic_manipulator", "autonomous_vehicle", "building_hvac"]


# --------------------------------------------------------------------------- #
# Shared tiny fixtures (state_dim = 2, control_dim = 1, output_dim = 2).
# --------------------------------------------------------------------------- #
def _make_config() -> PITSMRASConfig:
    cfg = PITSMRASConfig()
    cfg.network = NetworkConfig(
        input_dim=2, hidden_dim=16, output_dim=2, lstm_layers=1,
        attention_heads=2, embedding_dim=8,
    )
    cfg.physics = PhysicsConfig(
        n_generalized_coords=1, hamiltonian_hidden=16, dissipation_hidden=8,
    )
    return cfg


def _make_pitnn(cfg: PITSMRASConfig) -> PITNN:
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


def _all_finite(series_dict: dict) -> None:
    for key, series in series_dict.items():
        if not isinstance(series, list):
            continue
        for value in series:
            if isinstance(value, float):
                assert math.isfinite(value), f"non-finite in {key!r}: {value!r}"


# --------------------------------------------------------------------------- #
# Phase-0 scaffold gates.
# --------------------------------------------------------------------------- #
def test_package_imports() -> None:
    """The top-level package imports and exposes its version."""
    pkg = importlib.import_module("pits_mras")
    assert pkg.__version__ == "0.4.2"


@pytest.mark.parametrize("script_name", _EXAMPLE_SCRIPTS)
def test_example_script_imports(script_name: str) -> None:
    """Each example script imports without error."""
    script_path = _EXAMPLES_DIR / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(
        f"_pits_example_{script_name}", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


# --------------------------------------------------------------------------- #
# Phase-6 acceptance gate (un-skipped).
# --------------------------------------------------------------------------- #
def test_full_forward_pass_no_crash() -> None:
    """Run 10 steps of RealtimeInferenceEngine with no exceptions / NaN."""
    from pits_mras.inference.realtime import RealtimeInferenceEngine

    cfg = _make_config()
    pitnn = _make_pitnn(cfg)
    ref_model = _make_ref_model()
    controller = _make_controller(ref_model)
    controller.setup_safety_filter()

    engine = RealtimeInferenceEngine(
        pitnn, controller, ref_model, horizon=10, device="cpu"
    )
    x_p = torch.zeros(2)
    r = torch.ones(1) * 0.1
    for step in range(10):
        out = engine.step(x_p, r, dt=0.01)
        for key in ("u_safe", "e", "v_hat", "h_cbf", "f_hat"):
            assert torch.isfinite(out[key]).all(), f"non-finite {key} at {step}"
        assert isinstance(out["cbf_active"], bool)


# --------------------------------------------------------------------------- #
# Phase-5 acceptance gate (un-skipped).
# --------------------------------------------------------------------------- #
def test_pretrain_one_epoch() -> None:
    """One pretrain epoch produces finite (non-NaN) losses."""
    from pits_mras.training import pretrain_pitnn

    cfg = _make_config()
    pitnn = _make_pitnn(cfg)
    history = pretrain_pitnn(pitnn, cfg, epochs=1, batch_size=8, seed=0)
    assert isinstance(history, dict)
    assert len(history["total_loss"]) == 1
    _all_finite(history)


def test_cotrain_one_episode() -> None:
    """One co-training episode produces finite (non-NaN) losses."""
    from pits_mras.training import cotraining_loop

    cfg = _make_config()
    pitnn = _make_pitnn(cfg)
    ref_model = _make_ref_model()
    controller = _make_controller(ref_model)

    p_before = controller.critic.W_c.weight.detach().clone()
    metrics = cotraining_loop(
        pitnn, controller, ref_model, cfg,
        n_episodes=1, n_steps=8, batch_size=4, irl_window=3, seed=0,
    )
    assert isinstance(metrics, dict)
    _all_finite(metrics)
    p_after = controller.critic.W_c.weight.detach()
    assert not torch.allclose(p_before, p_after), "critic params did not change"
