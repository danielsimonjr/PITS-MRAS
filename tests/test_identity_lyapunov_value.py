"""Identity test (IP §11.1): Lyapunov = Value Function (Identity 1).

Targets ``pits_mras.utils.lyapunov`` and ``pits_mras.models.critic``.
Owning phase: Phase 1 / Phase 2 per ROADMAP.md (authored alongside its target
phase; §11 catalogs it under "Phase 8").

Verbatim mandated test names (ARCHITECTURE.md §7.3 / IP §11.1):
``test_kleinman_converges_to_care``, ``test_irl_critic_converges_to_lyapunov_P``,
``test_quadratic_basis_reconstructs_P``. The two Phase-1 tests are implemented
against ``utils/lyapunov.py``; the Phase-2 critic test stays skipped.
"""

import numpy as np
import torch

from pits_mras.controllers.reference_models import LinearReferenceModel
from pits_mras.models.critic import QuadraticCritic
from pits_mras.training.irl_trainer import train_irl_critic
from pits_mras.utils.lyapunov import (
    kleinman_iteration,
    quadratic_basis,
    solve_care,
)


def test_kleinman_converges_to_care() -> None:
    """Kleinman iteration converges to the CARE solution."""
    A = np.array([[0.0, 1.0], [-1.0, -1.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = np.eye(1)

    P_kleinman, K_kleinman = kleinman_iteration(A, B, Q, R)
    P_care, K_care = solve_care(A, B, Q, R)

    assert np.allclose(P_kleinman, P_care, atol=1e-6)
    assert np.allclose(K_kleinman, K_care, atol=1e-6)


def test_irl_critic_converges_to_lyapunov_P() -> None:
    """The IRL critic learns P_hat -> the Lyapunov/LQR P (Identity 1).

    Trains a :class:`QuadraticCritic` from synthetic optimal-closed-loop
    trajectories via the Phase-5 integral-RL trainer and asserts the recovered
    ``P_hat`` matches the CARE/LQR ``P_opt`` to within the trainer's own
    relative-Frobenius stop criterion (``< 0.01``). Also checks that the
    critic's in-place ``extract_P()`` agrees with ``P_opt``.
    """
    A_m = np.array([[0.0, 1.0], [-1.0, -1.0]])
    B = np.array([[0.0], [1.0]])
    C = np.eye(2)
    Q = np.eye(2)
    R = np.eye(1)

    ref = LinearReferenceModel(A_m=A_m, B_m=B, C_m=C, Q=Q, R=R)
    critic = QuadraticCritic(state_dim=2)

    P_hat, converged, n_iters = train_irl_critic(critic, ref, tol=0.01, seed=0)

    P_opt = torch.as_tensor(np.asarray(ref.P_opt.tolist()), dtype=P_hat.dtype)
    rel_err = float(torch.linalg.norm(P_hat - P_opt) / torch.linalg.norm(P_opt))

    assert converged
    assert n_iters >= 1
    # The deepest Identity-1 claim: the learned value matrix IS the CARE P.
    assert rel_err < 0.01

    # The fitted P is written into the critic in place; it must agree too.
    extracted = critic.extract_P()
    extracted_rel = float(torch.linalg.norm(extracted - P_opt) / torch.linalg.norm(P_opt))
    assert extracted_rel < 0.01
    assert torch.allclose(extracted, P_hat)


def test_quadratic_basis_reconstructs_P() -> None:
    """The quadratic Kronecker basis exactly reconstructs a given P."""
    # A known symmetric P, and a batch of points e.
    P = np.array([[2.0, 0.5], [0.5, 3.0]])
    n = 2
    rng = np.random.default_rng(0)
    e_np = rng.standard_normal((5, n))
    e = torch.tensor(e_np, dtype=torch.float64)

    phi = quadratic_basis(e)  # [batch, n*(n+1)//2]
    assert phi.shape == (5, n * (n + 1) // 2)

    # Build weight vector w such that wᵀφ(e) = eᵀPe.
    # Ordering of basis: [e1², e1·e2, e2²].
    # eᵀPe = P00 e1² + 2 P01 e1 e2 + P11 e2².
    w = torch.tensor([P[0, 0], 2.0 * P[0, 1], P[1, 1]], dtype=torch.float64)
    v_basis = phi @ w  # [batch]

    v_true = torch.tensor([e_np[k] @ P @ e_np[k] for k in range(5)], dtype=torch.float64)
    assert torch.allclose(v_basis, v_true, atol=1e-10)
