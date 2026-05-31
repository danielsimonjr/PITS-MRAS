"""Controller unit tests (Phase 4): reference model + MRAS controller.

Targets ``pits_mras.controllers.reference_models`` (``LinearReferenceModel``)
and ``pits_mras.controllers.mras`` (``MRASController``). The CBF safety-filter
tests live in ``tests/test_safety.py``; the LQR-warm-start costate identity
lives in ``tests/test_identity_costate.py``.
"""

import numpy as np
import pytest
import torch

from pits_mras.controllers.mras import MRASController
from pits_mras.controllers.reference_models import LinearReferenceModel
from pits_mras.utils.lyapunov import solve_care


def _make_ref_model() -> LinearReferenceModel:
    """A small Hurwitz reference model with matching B for state/control dims."""
    A_m = np.array([[-1.0, 0.5], [0.0, -2.0]])
    B_m = np.array([[1.0, 0.0], [0.0, 1.0]])
    C_m = np.eye(2)
    Q = np.eye(2)
    R = np.eye(2)
    return LinearReferenceModel(A_m, B_m, C_m, Q, R)


def test_reference_model_rejects_non_hurwitz() -> None:
    """A non-Hurwitz A_m must raise at construction time."""
    A_unstable = np.array([[1.0, 0.0], [0.0, -1.0]])  # eigenvalue +1
    B_m = np.eye(2)
    C_m = np.eye(2)
    with pytest.raises(ValueError):
        LinearReferenceModel(A_unstable, B_m, C_m, np.eye(2), np.eye(2))


def test_reference_model_buffers_and_dtype() -> None:
    """Buffers A_m/B_m/C_m/P/P_opt/K_opt/R_inv exist and are float32."""
    rm = _make_ref_model()
    for name in ("A_m", "B_m", "C_m", "Q", "R", "R_inv", "P", "P_opt", "K_opt"):
        buf = getattr(rm, name)
        assert isinstance(buf, torch.Tensor)
        assert buf.dtype == torch.float32

    # P solves A_m^T P + P A_m = -Q.
    A = rm.A_m
    P = rm.P
    resid = A.T @ P + P @ A + rm.Q
    assert torch.allclose(resid, torch.zeros_like(resid), atol=1e-4)

    # K_opt matches solve_care to within tolerance (Kleinman == CARE).
    P_care, K_care = solve_care(
        rm.A_m.numpy(), rm.B_m.numpy(), rm.Q.numpy(), rm.R.numpy()
    )
    assert np.allclose(rm.K_opt.numpy(), K_care, atol=1e-4)
    assert np.allclose(rm.P_opt.numpy(), P_care, atol=1e-4)


def test_reference_model_step_integrates() -> None:
    """One Euler step matches x + (A_m x + B_m r) dt exactly."""
    rm = _make_ref_model()
    x_m = torch.tensor([[1.0, -1.0]])
    r = torch.tensor([[0.5, 0.5]])
    dt = 0.01
    nxt = rm.step(x_m, r, dt)
    expected = x_m + (x_m @ rm.A_m.T + r @ rm.B_m.T) * dt
    assert torch.allclose(nxt, expected, atol=1e-6)

    # Free response (r=0) of a Hurwitz system shrinks the state norm.
    x = torch.tensor([[3.0, 2.0]])
    zero_r = torch.zeros(1, 2)
    for _ in range(200):
        x = rm.step(x, zero_r, 0.01)
    assert x.norm().item() < torch.tensor([[3.0, 2.0]]).norm().item()


def test_mras_forward_shapes() -> None:
    """MRASController.forward returns the documented dict with correct shapes."""
    rm = _make_ref_model()
    ctrl = MRASController(
        reference_model=rm,
        state_dim=2,
        control_dim=2,
        ref_dim=2,
        plant_dim=2,
        use_safety_filter=True,
    )
    ctrl.setup_safety_filter(safety_margin=10.0, decay_rate=1.0)
    batch = 5
    e = torch.randn(batch, 2)
    r = torch.randn(batch, 2)
    x_plant = torch.randn(batch, 2)
    out = ctrl(e, r, x_plant, apply_safety=True)
    assert out["u_nom"].shape == (batch, 2)
    assert out["u"].shape == (batch, 2)
    assert out["h_cbf"].shape == (batch,)
    assert out["slack"].shape == (batch,)

    # Without safety, u == u_nom and no CBF keys.
    out2 = ctrl(e, r, x_plant, apply_safety=False)
    assert torch.allclose(out2["u"], out2["u_nom"])
    assert "h_cbf" not in out2


def test_mras_nominal_control_law() -> None:
    """u_nom = -K_fb e + K_ff r + compensator(x_plant)."""
    rm = _make_ref_model()
    ctrl = MRASController(rm, 2, 2, 2, 2, use_safety_filter=False)
    e = torch.randn(3, 2)
    r = torch.randn(3, 2)
    x_plant = torch.randn(3, 2)
    out = ctrl(e, r, x_plant, apply_safety=False)
    expected = (
        -e @ ctrl.K_fb.T + r @ ctrl.K_ff.T + ctrl.compensator(x_plant)
    )
    assert torch.allclose(out["u_nom"], expected, atol=1e-6)


def test_mras_lqr_warm_start_sets_K_fb() -> None:
    """lqr_warm_start sets K_fb to solve_care's K and returns (P, K)."""
    rm = _make_ref_model()
    ctrl = MRASController(rm, 2, 2, 2, 2)
    Q = torch.eye(2)
    R = torch.eye(2)
    P_t, K_t = ctrl.lqr_warm_start(Q, R)
    P_care, K_care = solve_care(
        rm.A_m.numpy(), rm.B_m.numpy(), Q.numpy(), R.numpy()
    )
    assert np.allclose(ctrl.K_fb.numpy(), K_care, atol=1e-4)
    assert np.allclose(K_t.numpy(), K_care, atol=1e-4)
    assert np.allclose(P_t.numpy(), P_care, atol=1e-4)
