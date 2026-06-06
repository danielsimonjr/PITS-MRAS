r"""First-order dynamics linearization + the min-max-from-dynamics bridge.

Targets :func:`pits_mras.utils.linearization.linearize_dynamics` (ROADMAP
integration #6) and the
:func:`pits_mras.training.hinf_minmax.hinf_minmax_from_dynamics` adapter that
wires the neural H-infinity min-max loop to ANY dynamics callable.

Three claims:
  * exactness on an affine ``f`` (Jacobians equal the defining matrices),
  * correctness on a nonlinear ``f`` (Jacobian matches a hand/FD reference),
  * the bridge recovers the GARE oracle just like ``hinf_minmax_train`` does
    directly, and a Koopman ``latent_step`` linearizes to ``(A_z, B_z)``.
"""

import numpy as np
import torch

from pits_mras.models.koopman import KoopmanLiftingModel
from pits_mras.training.hinf_minmax import hinf_minmax_from_dynamics
from pits_mras.utils.linearization import linearize_dynamics


# --------------------------------------------------------------------------- #
# linearize_dynamics: exactness on affine f.
# --------------------------------------------------------------------------- #
def test_linearize_affine_recovers_matrices_exactly() -> None:
    """For f(x, u) = A0 x + B0 u (+ const), the Jacobians equal (A0, B0)."""
    A0 = torch.tensor([[0.0, 1.0], [-2.0, -3.0]], dtype=torch.float64)
    B0 = torch.tensor([[0.0], [1.5]], dtype=torch.float64)
    c = torch.tensor([0.3, -0.7], dtype=torch.float64)

    def f(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return A0 @ x + B0 @ u + c

    # Operating point is irrelevant for an affine f -- pick a non-origin point.
    x0 = torch.tensor([0.4, -1.1], dtype=torch.float64)
    u0 = torch.tensor([0.9], dtype=torch.float64)

    A, B = linearize_dynamics(f, x0, u0)
    assert A.shape == (2, 2)
    assert B.shape == (2, 1)
    assert torch.allclose(A, A0, atol=1e-10)
    assert torch.allclose(B, B0, atol=1e-10)


# --------------------------------------------------------------------------- #
# linearize_dynamics: correctness on a nonlinear f (pendulum-like).
# --------------------------------------------------------------------------- #
def test_linearize_nonlinear_matches_hand_jacobian() -> None:
    r"""Pendulum-like f: xdot = [x1, u - sin(x0) - x1].

    Hand Jacobians at (x0, u0):
        df/dx = [[0, 1], [-cos(x0_0), -1]],  df/du = [[0], [1]].
    """

    def f(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x0c, x1c = x[0], x[1]
        return torch.stack([x1c, u[0] - torch.sin(x0c) - x1c])

    x0 = torch.tensor([0.6, -0.2], dtype=torch.float64)
    u0 = torch.tensor([0.5], dtype=torch.float64)

    A, B = linearize_dynamics(f, x0, u0)

    A_hand = torch.tensor([[0.0, 1.0], [-float(torch.cos(x0[0])), -1.0]], dtype=torch.float64)
    B_hand = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    assert torch.allclose(A, A_hand, atol=1e-10)
    assert torch.allclose(B, B_hand, atol=1e-10)


def test_linearize_nonlinear_matches_finite_difference() -> None:
    """The analytic (jacrev) Jacobian matches a central finite-difference one."""

    def f(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.stack([x[0] * x[1], torch.cos(x[0]) + u[0] ** 2 - x[1]])

    x0 = torch.tensor([0.3, 1.2], dtype=torch.float64)
    u0 = torch.tensor([0.7], dtype=torch.float64)
    A, B = linearize_dynamics(f, x0, u0)

    eps = 1e-6
    n, m = 2, 1
    A_fd = torch.zeros(n, n, dtype=torch.float64)
    for j in range(n):
        dx = torch.zeros(n, dtype=torch.float64)
        dx[j] = eps
        A_fd[:, j] = (f(x0 + dx, u0) - f(x0 - dx, u0)) / (2 * eps)
    B_fd = torch.zeros(n, m, dtype=torch.float64)
    for j in range(m):
        du = torch.zeros(m, dtype=torch.float64)
        du[j] = eps
        B_fd[:, j] = (f(x0, u0 + du) - f(x0, u0 - du)) / (2 * eps)

    assert torch.allclose(A, A_fd, atol=1e-6)
    assert torch.allclose(B, B_fd, atol=1e-6)


# --------------------------------------------------------------------------- #
# linearize_dynamics: shape validation.
# --------------------------------------------------------------------------- #
def test_linearize_rejects_batched_inputs() -> None:
    """x0 / u0 must be 1-D single state/control vectors."""

    def f(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x + 0.0 * u.sum()

    x_bad = torch.zeros(3, 2)  # 2-D -> batched, not allowed
    u_ok = torch.zeros(1)
    try:
        linearize_dynamics(f, x_bad, u_ok)
        raise AssertionError("expected ValueError for 2-D x0")
    except ValueError as exc:
        assert "1-D" in str(exc)

    x_ok = torch.zeros(2)
    u_bad = torch.zeros(1, 1)
    try:
        linearize_dynamics(f, x_ok, u_bad)
        raise AssertionError("expected ValueError for 2-D u0")
    except ValueError as exc:
        assert "1-D" in str(exc)


def test_linearize_detects_wrong_output_dim() -> None:
    """A dynamics_fn that returns the wrong xdot size is rejected with a shape error."""

    def f_wrong(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Returns [state_dim + 1], not [state_dim].
        return torch.cat([x, u])

    x0 = torch.zeros(2, dtype=torch.float64)
    u0 = torch.zeros(1, dtype=torch.float64)
    try:
        linearize_dynamics(f_wrong, x0, u0)
        raise AssertionError("expected ValueError for wrong xdot dim")
    except ValueError as exc:
        assert "df/dx" in str(exc) or "df/du" in str(exc)


# --------------------------------------------------------------------------- #
# hinf_minmax_from_dynamics: recovers the GARE oracle on a LINEAR dynamics_fn.
# --------------------------------------------------------------------------- #
def test_from_dynamics_recovers_gare_oracle_linear() -> None:
    """A LINEAR dynamics_fn linearizes exactly, so the bridge converges to P*/K*.

    Reuses the small n=2 system + loose-but-meaningful thresholds from the
    existing min-max test. A LINEAR f means linearization is exact, so the only
    question is whether the (unchanged) min-max loop converges -- which it does.
    """
    A0 = torch.tensor([[0.0, 1.0], [-1.0, -1.0]], dtype=torch.float32)
    B0 = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

    def f(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return A0 @ x + B0 @ u

    Q = np.eye(2)
    R = np.eye(1)
    gamma = 5.0
    x0 = torch.zeros(2, dtype=torch.float32)
    u0 = torch.zeros(1, dtype=torch.float32)

    out = hinf_minmax_from_dynamics(f, x0, u0, Q, R, gamma, n_iters=3000, batch_size=256, seed=0)

    # (a) extracted (A, B) equal the linear system's matrices.
    assert np.allclose(out["A"], A0.numpy(), atol=1e-6)
    assert np.allclose(out["B"], B0.numpy(), atol=1e-6)

    # (b) oracle recovery: final P/K close to the analytic GARE solution.
    p_rel = np.linalg.norm(out["P_hat"] - out["P_star"]) / np.linalg.norm(out["P_star"])
    k_rel = np.linalg.norm(out["K_hat"] - out["K_star"]) / np.linalg.norm(out["K_star"])
    assert p_rel < 0.15, f"P_hat did not recover P*: rel dist {p_rel:.3f}"
    assert k_rel < 0.15, f"K_hat did not recover K*: rel dist {k_rel:.3f}"

    # No divergence.
    assert all(np.isfinite(out["P_dist"]))


# --------------------------------------------------------------------------- #
# Koopman bridge smoke test: latent_step linearizes to (A_z, B_z).
# --------------------------------------------------------------------------- #
def test_koopman_latent_step_linearizes_to_latent_matrices() -> None:
    """linearize_dynamics on a Koopman latent_step recovers (A_z, B_z) exactly.

    ``latent_step`` is z @ A_z^T + u @ B_z^T -- EXACTLY linear in (z, u) -- so its
    Jacobians are (A_z, B_z) regardless of the operating point. This demonstrates
    the Koopman -> robust-min-max wiring (treat the latent z as the "state").
    """
    torch.manual_seed(0)
    model = KoopmanLiftingModel(state_dim=2, control_dim=1, latent_dim=4)
    # Perturb A_z / B_z away from the warm-start init so the test is non-trivial.
    with torch.no_grad():
        model.A_z.add_(0.1 * torch.randn_like(model.A_z))
        model.B_z.add_(0.1 * torch.randn_like(model.B_z))
    A_z, B_z = model.latent_matrices()

    def latent_dyn(z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Single (unbatched) latent step: z @ A_z^T + u @ B_z^T == A_z z + B_z u.
        return model.latent_step(z.unsqueeze(0), u.unsqueeze(0)).squeeze(0)

    z0 = torch.zeros(4)  # operating point at the latent origin
    u0 = torch.zeros(1)
    A, B = linearize_dynamics(latent_dyn, z0, u0)

    assert A.shape == (4, 4)
    assert B.shape == (4, 1)
    assert torch.allclose(A, A_z.detach(), atol=1e-6)
    assert torch.allclose(B, B_z.detach(), atol=1e-6)
