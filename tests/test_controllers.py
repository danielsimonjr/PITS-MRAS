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


def test_mras_feedback_routed_through_costate_head() -> None:
    """Audit #2: the nominal feedback IS the costate-head optimal control.

    u_nom = costate_head(e).u_opt + K_ff r + compensator(x_plant), and the
    controller exposes ``lambda_hat`` / ``v_hat`` (IP §7.3).
    """
    rm = _make_ref_model()
    ctrl = MRASController(rm, 2, 2, 2, 2, use_safety_filter=False)
    e = torch.randn(4, 2)
    r = torch.randn(4, 2)
    x_plant = torch.randn(4, 2)
    out = ctrl(e, r, x_plant, apply_safety=False)
    _, u_fb = ctrl.costate_head(e)
    expected = u_fb + r @ ctrl.K_ff.T + ctrl.compensator(x_plant)
    assert torch.allclose(out["u_nom"], expected, atol=1e-6)
    assert "lambda_hat" in out and "v_hat" in out


def test_mras_feedback_equals_lqr_at_warmstart_and_tracks_critic() -> None:
    """Audit #2/#4: feedback equals -K_opt e at init (critic warm-started to
    P_opt) and then TRACKS the learned critic, not a frozen K_fb buffer."""
    rm = _make_ref_model()
    ctrl = MRASController(rm, 2, 2, 2, 2, use_safety_filter=False)
    e = torch.randn(5, 2)

    # At construction the critic is warm-started to P_opt, so the costate-head
    # feedback equals the LQR gain -K_opt e (== -e @ K_fb.T).
    _, u_fb0 = ctrl.costate_head(e)
    assert torch.allclose(u_fb0, -e @ ctrl.K_fb.T, atol=1e-5)

    # Change the critic's P; feedback must follow the critic (-R^-1 B^T P e),
    # NOT remain pinned to the frozen K_fb.
    P_new = torch.tensor([[3.0, 0.5], [0.5, 5.0]])
    ctrl.critic.set_P(P_new)
    _, u_fb1 = ctrl.costate_head(e)
    expected1 = -(e @ P_new @ rm.B_m) @ rm.R_inv.T
    assert torch.allclose(u_fb1, expected1, atol=1e-5)
    assert not torch.allclose(u_fb1, -e @ ctrl.K_fb.T, atol=1e-3)


def test_mras_regressor_concatenates_e_r_xp() -> None:
    """Audit #3: phi_c = [e, r, x_p] (IP §7.3)."""
    rm = _make_ref_model()
    ctrl = MRASController(rm, 2, 2, 2, 2, use_safety_filter=False)
    e = torch.randn(4, 2)
    r = torch.randn(4, 2)
    x_p = torch.randn(4, 2)
    phi = ctrl.mras_regressor(e, r, x_p)
    assert phi.shape == (4, 6)
    assert torch.allclose(phi, torch.cat([e, r, x_p], dim=-1))


def test_dpg_action_gradient_zero_at_optimum() -> None:
    """Audit #3: the DPG action-value gradient Ru + B^T P_hat e vanishes at the
    optimal control u* = -R^-1 B^T P_hat e (backward-compatible LQR limit)."""
    rm = _make_ref_model()
    ctrl = MRASController(rm, 2, 2, 2, 2, use_safety_filter=False)  # critic ~ P_opt
    e = torch.randn(5, 2)
    _, u_star = ctrl.costate_head(e)  # optimal control for the warm-started critic
    q_grad = ctrl.dpg_action_value_gradient(e, u_star)
    assert q_grad.shape == (5, 2)
    assert torch.allclose(q_grad, torch.zeros_like(q_grad), atol=1e-5)
    # Away from the optimum it is non-zero.
    q_grad_off = ctrl.dpg_action_value_gradient(e, u_star + 1.0)
    assert q_grad_off.abs().max() > 1e-3


def test_dpg_actor_step_updates_actor_not_critic() -> None:
    """Audit #3: the DPG step accumulates gradients on the actor params
    (K_ff, compensator) and leaves the critic (trained by IRL) untouched."""
    rm = _make_ref_model()
    ctrl = MRASController(rm, 2, 2, 2, 2, use_safety_filter=False)
    e = torch.randn(6, 2)
    r = torch.randn(6, 2)
    x_p = torch.randn(6, 2)
    ctrl.zero_grad(set_to_none=True)
    ctrl.dpg_actor_step(e, r, x_p, gamma_c=0.1)
    assert ctrl.K_ff.grad is not None and torch.isfinite(ctrl.K_ff.grad).all()
    comp_grads = [p.grad for p in ctrl.compensator.parameters() if p.grad is not None]
    assert len(comp_grads) > 0 and all(torch.isfinite(g).all() for g in comp_grads)
    # Critic is the IRL-trained value function: DPG must not write to it.
    assert ctrl.critic.W_c.weight.grad is None


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
