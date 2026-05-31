"""Phase-6 inference-engine tests (IP §9).

Covers:
  * ``RealtimeInferenceEngine.step`` returns the documented dict with correct
    shapes and finite values.
  * The bounded history deques stay capped at ``horizon``.
  * A 10-step closed-loop run is all-finite (the §11.6 smoke-gate behavior,
    exercised here at module scope as well).
  * ``cbf_active`` is a plain ``bool``.
  * ``ParallelInferenceEngine`` starts and stops cleanly (threads join, no
    deadlock) within a short timeout.
"""

import math

import numpy as np
import torch

from pits_mras.config import NetworkConfig, PhysicsConfig, PITSMRASConfig
from pits_mras.controllers.mras import MRASController
from pits_mras.controllers.reference_models import LinearReferenceModel
from pits_mras.inference.parallel import ControllerState, ParallelInferenceEngine
from pits_mras.inference.realtime import RealtimeInferenceEngine
from pits_mras.models import PITNN

_HORIZON = 5


# --------------------------------------------------------------------------- #
# Tiny fixtures (state_dim = 2, control_dim = 1, output_dim = 2).
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


def _make_ref_model() -> LinearReferenceModel:
    A_m = np.array([[0.0, 1.0], [-1.0, -1.0]])
    B_m = np.array([[0.0], [1.0]])
    C_m = np.eye(2)
    Q = np.eye(2)
    R = np.eye(1)
    return LinearReferenceModel(A_m, B_m, C_m, Q, R)


def _make_engine() -> RealtimeInferenceEngine:
    cfg = _make_config()
    pitnn = PITNN(cfg.network, cfg.physics)
    ref_model = _make_ref_model()
    controller = MRASController(
        reference_model=ref_model, state_dim=2, control_dim=1, ref_dim=1,
        plant_dim=2, use_safety_filter=True,
    )
    controller.setup_safety_filter()
    return RealtimeInferenceEngine(
        pitnn, controller, ref_model, horizon=_HORIZON, device="cpu"
    )


# --------------------------------------------------------------------------- #
# RealtimeInferenceEngine.
# --------------------------------------------------------------------------- #
def test_step_returns_documented_dict_with_shapes() -> None:
    engine = _make_engine()
    x_p = torch.zeros(2)
    r = torch.ones(1)
    out = engine.step(x_p, r, dt=0.01)

    for key in ("u_safe", "e", "v_hat", "h_cbf", "f_hat", "cbf_active"):
        assert key in out, f"missing key {key!r}"

    assert out["u_safe"].shape == (1,)       # control_dim
    assert out["e"].shape == (2,)            # state_dim
    assert out["f_hat"].shape == (2,)        # output_dim
    assert out["v_hat"].shape == ()          # scalar value
    assert out["h_cbf"].shape == ()          # scalar CBF value


def test_cbf_active_is_bool() -> None:
    engine = _make_engine()
    out = engine.step(torch.zeros(2), torch.ones(1))
    assert isinstance(out["cbf_active"], bool)


def test_history_deques_bounded_at_horizon() -> None:
    engine = _make_engine()
    for _ in range(3 * _HORIZON):
        engine.step(torch.zeros(2), torch.ones(1))
    assert len(engine._x_hist) == _HORIZON
    assert len(engine._u_hist) == _HORIZON
    assert len(engine._e_hist) == _HORIZON


def test_ten_step_run_all_finite() -> None:
    engine = _make_engine()
    for k in range(10):
        out = engine.step(torch.zeros(2), torch.ones(1) * 0.1)
        for key in ("u_safe", "e", "v_hat", "h_cbf", "f_hat"):
            assert torch.isfinite(out[key]).all(), f"non-finite {key} at step {k}"


# --------------------------------------------------------------------------- #
# ParallelInferenceEngine (reference skeleton).
# --------------------------------------------------------------------------- #
def test_controller_state_dataclass_defaults() -> None:
    state = ControllerState()
    assert isinstance(state.cbf_activations, int)
    assert isinstance(state.step_count, int)
    assert math.isfinite(state.v_hat)


def test_parallel_engine_starts_and_stops_cleanly() -> None:
    engine = _make_engine()
    par = ParallelInferenceEngine(
        engine, x_p=torch.zeros(2), r=torch.ones(1) * 0.1,
    )
    par.start()
    # Let the threads tick a few cycles.
    par.wait(0.2)
    par.stop(timeout=2.0)
    assert not par.is_alive(), "threads failed to join within timeout"
    # The control thread should have advanced the shared step counter.
    assert par.state.step_count > 0


def test_parallel_engine_double_stop_is_safe() -> None:
    engine = _make_engine()
    par = ParallelInferenceEngine(engine, x_p=torch.zeros(2), r=torch.zeros(1))
    par.start()
    par.stop(timeout=2.0)
    par.stop(timeout=2.0)  # idempotent, must not raise
    assert not par.is_alive()
