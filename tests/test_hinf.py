"""H∞ robust-control core: the GARE solver and the analytic adversary head.

Targets ``pits_mras.utils.lyapunov.solve_gare`` and
``pits_mras.models.critic.AdversaryHead``. Verifies the GARE is satisfied, that
``gamma -> inf`` recovers the CARE, that an infeasible ``gamma`` raises, that the
disturbance matrix ``D`` defaults to ``B``, and that the adversary head equals
the analytic worst-case disturbance ``w* = gamma^-2 D^T P e`` by construction.
"""

import numpy as np
import pytest
import torch

from pits_mras.models.critic import AdversaryHead, QuadraticCritic
from pits_mras.utils.lyapunov import solve_care, solve_gare


def _system() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A small Hurwitz LTI test system (A stable, single input)."""
    A = np.array([[0.0, 1.0], [-1.0, -1.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = np.eye(1)
    return A, B, Q, R


def _gare_residual(A, B, Q, R, D, gamma, P) -> float:
    M = B @ np.linalg.inv(R) @ B.T - (1.0 / gamma**2) * D @ D.T
    return float(np.linalg.norm(A.T @ P + P @ A + Q - P @ M @ P))


def test_solve_gare_satisfies_gare() -> None:
    """The returned P solves the GARE, is symmetric PD, and the worst-case
    closed loop A - M P is Hurwitz."""
    A, B, Q, R = _system()
    gamma = 5.0
    P, K, L = solve_gare(A, B, Q, R, gamma)
    assert _gare_residual(A, B, Q, R, B, gamma, P) < 1e-8
    assert np.allclose(P, P.T, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(P)) > 0
    M = B @ np.linalg.inv(R) @ B.T - (1.0 / gamma**2) * B @ B.T
    assert np.max(np.real(np.linalg.eigvals(A - M @ P))) < 0
    # Gains: K = R^-1 B^T P, L = gamma^-2 D^T P.
    assert np.allclose(K, np.linalg.solve(R, B.T @ P), atol=1e-10)
    assert np.allclose(L, (1.0 / gamma**2) * B.T @ P, atol=1e-10)


def test_solve_gare_large_gamma_recovers_care() -> None:
    """As gamma -> inf the GARE reduces to the CARE (the disturbance term vanishes)."""
    A, B, Q, R = _system()
    P_gare, _, _ = solve_gare(A, B, Q, R, gamma=1e6)
    P_care, _ = solve_care(A, B, Q, R)
    assert np.allclose(P_gare, P_care, atol=1e-3)


def test_solve_gare_raises_on_infeasible_gamma() -> None:
    """A gamma below the H-infinity-achievable bound has no stabilizing PD
    solution -> ValueError."""
    A, B, Q, R = _system()
    with pytest.raises(ValueError):
        solve_gare(A, B, Q, R, gamma=0.1)


def test_solve_gare_default_D_is_B() -> None:
    """Omitting D is equivalent to passing D = B (matched disturbance)."""
    A, B, Q, R = _system()
    P_default, _, _ = solve_gare(A, B, Q, R, gamma=5.0)
    P_explicit, _, _ = solve_gare(A, B, Q, R, gamma=5.0, D=B)
    assert np.allclose(P_default, P_explicit, atol=1e-12)


def test_adversary_head_equals_analytic_worst_case() -> None:
    """The adversary head returns w* = gamma^-2 D^T P e (== (1/2gamma^2) D^T grad V),
    by construction from the critic gradient, for a critic warm-started to the
    GARE P."""
    A, B, Q, R = _system()
    gamma = 5.0
    P, _, _ = solve_gare(A, B, Q, R, gamma)
    critic = QuadraticCritic(state_dim=2)
    critic.set_P(torch.tensor(P, dtype=torch.float32))
    D = torch.tensor(B, dtype=torch.float32)  # [state_dim, dist_dim]
    head = AdversaryHead(critic, D, gamma)

    e = torch.tensor([[1.0, -0.5], [0.2, 0.3]], dtype=torch.float32)
    w = head(e)
    assert w.shape == (2, 1)  # [batch, dist_dim]
    # Analytic: w*[b] = gamma^-2 D^T P e[b].
    P_t = torch.tensor(P, dtype=torch.float32)
    expected = (1.0 / gamma**2) * (e @ P_t @ D)  # [batch, dist_dim]
    assert torch.allclose(w, expected, atol=1e-5)
    # Equivalent to (1/2gamma^2) D^T grad V (grad V = 2 P e by construction).
    grad_v = critic.gradient(e.detach())
    assert torch.allclose(w, (1.0 / (2.0 * gamma**2)) * (grad_v @ D), atol=1e-5)
