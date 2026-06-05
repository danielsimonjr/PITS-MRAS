"""Identity test (IP §11.2): Costate = Critic Gradient (Identity 2).

Targets ``pits_mras.models.critic`` (``CostateHead`` / ``QuadraticCritic``).
Owning phase: Phase 2 per ROADMAP.md (authored alongside its target phase;
§11 catalogs it under "Phase 8").

Verbatim mandated test names (ARCHITECTURE.md §7.3 / IP §11.2):
``test_costate_equals_grad_V``, ``test_optimal_control_equals_lqr_gain``.

Phase-2 status:
- ``test_costate_equals_grad_V`` is UN-SKIPPED: the costate head's action IS the
  autodiff gradient of the critic by construction, so the identity holds with
  only Phase-2 code.

Phase-4 status:
- ``test_optimal_control_equals_lqr_gain`` is now UN-SKIPPED: after
  ``MRASController.lqr_warm_start(Q, R)`` the controller's nominal feedback
  ``u = -K_fb e`` equals the LQR optimal ``-K e`` (K from ``solve_care``), and
  (because Phase 4 added ``QuadraticCritic.set_P``) the critic costate-derived
  control ``-R^{-1} B^T (1/2 grad V) = -K e`` matches as well.
"""

import numpy as np
import torch

from pits_mras.controllers.mras import MRASController
from pits_mras.controllers.reference_models import LinearReferenceModel
from pits_mras.models.critic import CostateHead, QuadraticCritic
from pits_mras.utils.lyapunov import solve_care


def test_costate_equals_grad_V() -> None:
    """I2: the costate equals the autodiff gradient of the value head.

    The ``CostateHead`` returns ``lambda_hat = critic.gradient(e)``; this is, by
    construction, ``grad_e V(e)``. We verify it against a fresh autograd pass and
    against the closed form ``2 P e`` for the quadratic critic.
    """
    torch.manual_seed(0)
    n, m = 4, 2
    critic = QuadraticCritic(state_dim=n)
    head = CostateHead(critic, torch.eye(m), torch.randn(n, m))

    e = torch.randn(6, n)
    lam_head, _ = head(e)

    # Independent autograd gradient of V.
    e2 = e.clone().detach().requires_grad_(True)
    V = critic(e2)
    grad_V = torch.autograd.grad(V.sum(), e2)[0]
    assert torch.allclose(lam_head, grad_V, atol=1e-5)

    # Closed form 2 P e.
    P = critic.extract_P()
    assert torch.allclose(lam_head, 2.0 * (e @ P.T), atol=1e-5)


def test_optimal_control_equals_lqr_gain() -> None:
    """u* = -R^-1 B^T grad V recovers the LQR gain in the linear limit.

    After ``lqr_warm_start``, two things must hold numerically:
      1. The MRAS nominal feedback ``u = -K_fb e`` equals ``-K e`` (CARE gain).
      2. The critic costate control ``-R^{-1} B^T (1/2 grad V)`` equals ``-K e``.
         With ``V = e^T P e`` and ``set_P(P)``, grad V = 2 P e, so the costate
         (half-gradient) is ``P e`` and ``-R^{-1} B^T P e = -K e``.
    """
    A_m = np.array([[-1.0, 0.5], [0.0, -2.0]])
    B_m = np.array([[1.0, 0.0], [0.0, 1.0]])
    C_m = np.eye(2)
    Q_np = np.eye(2)
    R_np = np.eye(2)
    rm = LinearReferenceModel(A_m, B_m, C_m, Q_np, R_np)
    ctrl = MRASController(rm, state_dim=2, control_dim=2, ref_dim=2, plant_dim=2)

    Q = torch.tensor(Q_np, dtype=torch.float32)
    R = torch.tensor(R_np, dtype=torch.float32)
    P_t, K_t = ctrl.lqr_warm_start(Q, R)

    P_care, K_care = solve_care(A_m, B_m, Q_np, R_np)

    # (1) K_fb == CARE gain.
    assert np.allclose(ctrl.K_fb.numpy(), K_care, atol=1e-4)

    e = torch.randn(7, 2)
    u_fb = -e @ ctrl.K_fb.T
    u_lqr = -e @ torch.tensor(K_care, dtype=torch.float32).T
    assert torch.allclose(u_fb, u_lqr, atol=1e-4)

    # (2) Critic was warm-started so extract_P == P_care.
    P_hat = ctrl.critic.extract_P()
    assert torch.allclose(P_hat, torch.tensor(P_care, dtype=torch.float32), atol=1e-4)

    # Costate control: u* = -R^{-1} B^T * (1/2 grad V) = -R^{-1} B^T P e = -K e.
    R_inv = rm.R_inv
    B = rm.B_m
    grad_V = ctrl.critic.gradient(e)  # 2 P e
    costate = 0.5 * grad_V  # P e
    u_costate = -(costate @ B) @ R_inv.T
    assert torch.allclose(u_costate, u_lqr, atol=1e-4)


def test_costate_head_u_opt_equals_lqr_gain() -> None:
    """The ``CostateHead``'s OWN ``u_opt`` must equal the LQR optimal ``-Ke``.

    Regression for the missing factor-of-½. With ``V = e^T P e`` the costate is
    ``grad V = 2 P e``; the optimal control is ``u* = -½ R^{-1} B^T grad V =
    -R^{-1} B^T P e = -K e`` (PMP: ∂H/∂u = 2Ru + B^T λ = 0). The head must apply
    the same ½ convention as ``HJBResidualLoss(half_grad=True)`` -- prior to the
    fix it returned ``-2 K e``. ``lambda_hat`` itself stays the true costate.
    """
    A_m = np.array([[-1.0, 0.5], [0.0, -2.0]])
    B_m = np.array([[1.0, 0.0], [0.0, 1.0]])
    C_m = np.eye(2)
    Q_np = np.eye(2)
    R_np = np.eye(2)
    rm = LinearReferenceModel(A_m, B_m, C_m, Q_np, R_np)
    ctrl = MRASController(rm, state_dim=2, control_dim=2, ref_dim=2, plant_dim=2)

    Q = torch.tensor(Q_np, dtype=torch.float32)
    R = torch.tensor(R_np, dtype=torch.float32)
    ctrl.lqr_warm_start(Q, R)
    _, K_care = solve_care(A_m, B_m, Q_np, R_np)

    e = torch.randn(7, 2)
    u_lqr = -e @ torch.tensor(K_care, dtype=torch.float32).T

    lambda_hat, u_opt = ctrl.costate_head(e)
    # lambda_hat is the true costate grad V = 2 P e (unchanged by the ½ fix).
    assert torch.allclose(lambda_hat, ctrl.critic.gradient(e), atol=1e-5)
    # u_opt applies the ½ and recovers the LQR gain (was -2Ke before the fix).
    assert torch.allclose(u_opt, u_lqr, atol=1e-4)
