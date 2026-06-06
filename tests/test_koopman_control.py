"""Tests for the Koopman-LQR controller (ROADMAP integration #5).

Targets ``pits_mras.controllers.koopman_control.KoopmanLQRController``. ADDITIVE:
wires the deep Koopman lifting model into the control loop by solving the
Riccati problem on the *learned latent linear system* ``(A_z, B_z)`` and closing
the loop on lifted coordinates. Does NOT modify the Koopman model or the
analytic core.

Design choices that drive the deterministic asserts:
- A generic untrained Koopman model has ``A_z`` near identity with zeroed
  extra-coord blocks, which is NOT stabilizable enough for ``solve_care``. So the
  oracle / stability tests SET ``A_z, B_z`` to deliberately-stabilizable known
  systems (documented inline) rather than relying on random init.
- ``solve_care`` is the single source of truth for the gain; the controller just
  wraps it on the lifted coordinates, so ``K_z`` must match a direct
  ``solve_care`` call to tight tolerance.
"""

import numpy as np
import pytest
import torch

from pits_mras.controllers.koopman_control import KoopmanLQRController
from pits_mras.models.koopman import KoopmanLiftingModel
from pits_mras.utils.lyapunov import check_hurwitz, solve_care


def _stabilizable_model(
    state_dim: int = 2,
    control_dim: int = 1,
    latent_dim: int = 2,
    include_state: bool = True,
) -> KoopmanLiftingModel:
    """Build a Koopman model with a deliberately-stabilizable latent system.

    A generic untrained model has zeroed extra-coord rows in ``A_z`` (not
    stabilizable). We overwrite ``A_z, B_z`` with a known controllable pair so
    ``solve_care`` succeeds. The chosen ``A_z`` is unstable (eigenvalues with
    positive real part) so the Riccati solution is non-trivial, and ``B_z`` makes
    the pair controllable.
    """
    model = KoopmanLiftingModel(
        state_dim=state_dim,
        control_dim=control_dim,
        latent_dim=latent_dim,
        include_state=include_state,
    )
    rng = np.random.default_rng(0)
    # Unstable but controllable: A_z has +0.5 on the diagonal; B_z all-ones cols.
    A_z = 0.5 * np.eye(latent_dim) + 0.1 * rng.standard_normal((latent_dim, latent_dim))
    B_z = np.ones((latent_dim, control_dim)) + 0.05 * rng.standard_normal((latent_dim, control_dim))
    with torch.no_grad():
        model.A_z.copy_(torch.as_tensor(A_z, dtype=torch.float32))
        model.B_z.copy_(torch.as_tensor(B_z, dtype=torch.float32))
    return model


# --------------------------------------------------------------------------- #
# Oracle recovery / correctness
# --------------------------------------------------------------------------- #
def test_oracle_recovery_matches_solve_care() -> None:
    """Controller K_z must equal a direct solve_care on (A_z, B_z, Q_z, R)."""
    latent_dim, control_dim = 3, 1
    model = _stabilizable_model(state_dim=3, control_dim=control_dim, latent_dim=latent_dim)
    Q = np.eye(3)
    R = np.eye(control_dim) * 2.0
    ctrl = KoopmanLQRController(model, Q, R)

    A_z, B_z = (m.detach().cpu().numpy().astype(np.float64) for m in model.latent_matrices())
    # include_state=True with latent_dim == state_dim -> Q_z == Q embedded fully.
    Q_z = np.eye(latent_dim)
    _, K_z_oracle = solve_care(A_z, B_z, Q_z, R)

    K_z = ctrl.latent_gain().detach().cpu().numpy()
    assert K_z.shape == (control_dim, latent_dim)
    np.testing.assert_allclose(K_z, K_z_oracle, rtol=1e-4, atol=1e-5)


def test_latent_closed_loop_hurwitz() -> None:
    """A_z - B_z @ K_z must be Hurwitz (stabilizing solution)."""
    model = _stabilizable_model(state_dim=2, control_dim=1, latent_dim=2)
    ctrl = KoopmanLQRController(model, np.eye(2), np.eye(1))
    A_z, B_z = (m.detach().cpu().numpy().astype(np.float64) for m in model.latent_matrices())
    K_z = ctrl.latent_gain().detach().cpu().numpy()
    A_cl = A_z - B_z @ K_z
    assert check_hurwitz(A_cl)


def test_q_latent_override() -> None:
    """An explicit q_latent is used verbatim instead of the embedded Q."""
    latent_dim, control_dim = 4, 1
    model = _stabilizable_model(state_dim=2, control_dim=control_dim, latent_dim=latent_dim)
    Q = np.eye(2)  # ignored when q_latent given
    R = np.eye(control_dim)
    Q_z = np.diag([3.0, 1.0, 2.0, 0.5])
    ctrl = KoopmanLQRController(model, Q, R, q_latent=Q_z)

    A_z, B_z = (m.detach().cpu().numpy().astype(np.float64) for m in model.latent_matrices())
    _, K_z_oracle = solve_care(A_z, B_z, Q_z, R)
    np.testing.assert_allclose(
        ctrl.latent_gain().detach().cpu().numpy(), K_z_oracle, rtol=1e-4, atol=1e-5
    )


def test_q_z_embedding_convention() -> None:
    """Embedded Q_z puts Q in the leading state block and zeros elsewhere."""
    latent_dim = 4
    model = _stabilizable_model(state_dim=2, control_dim=1, latent_dim=latent_dim)
    Q = np.array([[2.0, 0.0], [0.0, 3.0]])
    ctrl = KoopmanLQRController(model, Q, np.eye(1))
    Q_z = ctrl.Q_z.detach().cpu().numpy()
    expected = np.zeros((latent_dim, latent_dim))
    expected[:2, :2] = Q
    np.testing.assert_allclose(Q_z, expected)


# --------------------------------------------------------------------------- #
# Lifted closed-loop stability (discrete Euler rollout)
# --------------------------------------------------------------------------- #
def test_lifted_error_decays() -> None:
    """Latent error under continuous closed loop A_z - B_z K_z decays to 0."""
    model = _stabilizable_model(state_dim=2, control_dim=1, latent_dim=2)
    ctrl = KoopmanLQRController(model, np.eye(2), np.eye(1))
    A_z, B_z = (m.detach().cpu().numpy().astype(np.float64) for m in model.latent_matrices())
    K_z = ctrl.latent_gain().detach().cpu().numpy()
    A_cl = A_z - B_z @ K_z  # Hurwitz

    dt = 1e-2
    z = np.array([1.0, -1.0])
    n0 = float(np.linalg.norm(z))
    norms = []
    for _ in range(5000):
        z = z + dt * (A_cl @ z)  # forward-Euler continuous closed loop
        norms.append(float(np.linalg.norm(z)))
    # A Hurwitz (possibly non-normal) closed loop may have a bounded transient,
    # but must decay to the origin: the tail is far below the start.
    assert norms[-1] < 1e-4 * n0
    assert max(norms[-100:]) < max(norms[:100])  # later window strictly smaller


# --------------------------------------------------------------------------- #
# Shapes / validation / broadcasting
# --------------------------------------------------------------------------- #
def test_control_output_shape() -> None:
    """control() returns [batch, control_dim]."""
    model = _stabilizable_model(state_dim=2, control_dim=1, latent_dim=2)
    ctrl = KoopmanLQRController(model, np.eye(2), np.eye(1))
    x = torch.randn(7, 2)
    x_ref = torch.randn(7, 2)
    u = ctrl.control(x, x_ref)
    assert u.shape == (7, 1)


def test_control_zero_at_reference() -> None:
    """When x == x_ref the lifted error is zero, so u == 0."""
    model = _stabilizable_model(state_dim=2, control_dim=1, latent_dim=2)
    ctrl = KoopmanLQRController(model, np.eye(2), np.eye(1))
    x = torch.randn(4, 2)
    u = ctrl.control(x, x)
    torch.testing.assert_close(u, torch.zeros(4, 1))


def test_control_broadcasts_reference() -> None:
    """A single reference row broadcasts across a batch of states."""
    model = _stabilizable_model(state_dim=2, control_dim=1, latent_dim=2)
    ctrl = KoopmanLQRController(model, np.eye(2), np.eye(1))
    x = torch.randn(5, 2)
    x_ref = torch.zeros(1, 2)
    u = ctrl.control(x, x_ref)
    assert u.shape == (5, 1)


def test_bad_q_shape_raises() -> None:
    """A Q whose shape does not match state_dim must raise."""
    model = _stabilizable_model(state_dim=2, control_dim=1, latent_dim=2)
    with pytest.raises(ValueError):
        KoopmanLQRController(model, np.eye(3), np.eye(1))


def test_bad_r_shape_raises() -> None:
    """An R whose shape does not match control_dim must raise."""
    model = _stabilizable_model(state_dim=2, control_dim=1, latent_dim=2)
    with pytest.raises(ValueError):
        KoopmanLQRController(model, np.eye(2), np.eye(2))


def test_bad_q_latent_shape_raises() -> None:
    """A q_latent whose shape does not match latent_dim must raise."""
    model = _stabilizable_model(state_dim=2, control_dim=1, latent_dim=4)
    with pytest.raises(ValueError):
        KoopmanLQRController(model, np.eye(2), np.eye(1), q_latent=np.eye(3))


def test_non_stabilizable_surfaces_error() -> None:
    """A generic untrained model (zeroed extra-coord rows) is not stabilizable.

    latent_dim > state_dim leaves the bottom rows of A_z zero and B_z zero, so
    those modes are uncontrollable at the origin -> solve_care must fail and the
    controller must surface that as an error.
    """
    model = KoopmanLiftingModel(state_dim=2, control_dim=1, latent_dim=4, include_state=True)
    with pytest.raises(Exception):
        KoopmanLQRController(model, np.eye(2), np.eye(1))
