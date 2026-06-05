r"""Differentiable CARE / GARE via implicit differentiation (ROADMAP #6).

Targets ``pits_mras.utils.lyapunov.differentiable_care`` and
``differentiable_gare``: thin wrappers around a ``torch.autograd.Function`` that
solve the Riccati equation with scipy (graph-breaking) in the forward pass under
``torch.no_grad`` and assemble exact gradients in the backward pass via the
implicit function theorem on the Riccati residual.

The gold standard here is ``torch.autograd.gradcheck`` in float64: it compares
the analytic backward against finite differences of the forward solve.

Symmetry handling: ``Q``, ``R`` (and ``D D^T``) enter the Riccati equation only
through symmetric combinations. To keep gradcheck on valid perturbation
directions, the wrappers symmetrize ``Q`` and ``R`` internally (``X -> (X+X^T)/2``)
so that an arbitrary unconstrained perturbation of those inputs maps through a
differentiable projection — gradcheck then exercises the full (unsymmetric)
input space and the analytic backward must match. ``A`` and ``B`` are
unconstrained.
"""

import numpy as np
import pytest
import torch

from pits_mras.utils.lyapunov import (
    differentiable_care,
    differentiable_gare,
    solve_care,
    solve_gare,
)


def _stabilizable_systems():
    """A few small stabilizable systems (n=2, 3) with SPD Q, R."""
    rng = np.random.default_rng(0)
    systems = []
    # n=2, single input (controllable double integrator-ish).
    systems.append(
        (
            np.array([[0.0, 1.0], [-1.0, -1.0]]),
            np.array([[0.0], [1.0]]),
            np.eye(2),
            np.eye(1),
        )
    )
    # n=2, two inputs, random SPD Q, R.
    for _ in range(2):
        n, m = 2, 2
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, m))
        Qh = rng.standard_normal((n, n))
        Q = Qh @ Qh.T + n * np.eye(n)
        Rh = rng.standard_normal((m, m))
        R = Rh @ Rh.T + m * np.eye(m)
        systems.append((A, B, Q, R))
    # n=3, two inputs.
    n, m = 3, 2
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, m))
    Qh = rng.standard_normal((n, n))
    Q = Qh @ Qh.T + n * np.eye(n)
    Rh = rng.standard_normal((m, m))
    R = Rh @ Rh.T + m * np.eye(m)
    systems.append((A, B, Q, R))
    return systems


# --------------------------------------------------------------------------- #
# Equivalence: differentiable solver value == scipy solver value.
# --------------------------------------------------------------------------- #
def test_differentiable_care_matches_solve_care() -> None:
    for A, B, Q, R in _stabilizable_systems():
        P_ref, _ = solve_care(A, B, Q, R)
        P = differentiable_care(
            torch.tensor(A, dtype=torch.float64),
            torch.tensor(B, dtype=torch.float64),
            torch.tensor(Q, dtype=torch.float64),
            torch.tensor(R, dtype=torch.float64),
        )
        assert torch.allclose(P.detach(), torch.tensor(P_ref), atol=1e-8)


def test_differentiable_gare_matches_solve_gare() -> None:
    for A, B, Q, R in _stabilizable_systems():
        gamma = 8.0
        P_ref, _, _ = solve_gare(A, B, Q, R, gamma)
        P = differentiable_gare(
            torch.tensor(A, dtype=torch.float64),
            torch.tensor(B, dtype=torch.float64),
            torch.tensor(Q, dtype=torch.float64),
            torch.tensor(R, dtype=torch.float64),
            gamma,
        )
        assert torch.allclose(P.detach(), torch.tensor(P_ref), atol=1e-7)


def test_differentiable_care_returns_symmetric_pd() -> None:
    A, B, Q, R = _stabilizable_systems()[0]
    P = differentiable_care(
        torch.tensor(A, dtype=torch.float64),
        torch.tensor(B, dtype=torch.float64),
        torch.tensor(Q, dtype=torch.float64),
        torch.tensor(R, dtype=torch.float64),
    ).detach()
    assert torch.allclose(P, P.T, atol=1e-9)
    assert torch.linalg.eigvalsh(P).min() > 0


# --------------------------------------------------------------------------- #
# gradcheck: the gold standard.
# --------------------------------------------------------------------------- #
def test_gradcheck_differentiable_care() -> None:
    for A, B, Q, R in _stabilizable_systems():
        At = torch.tensor(A, dtype=torch.float64, requires_grad=True)
        Bt = torch.tensor(B, dtype=torch.float64, requires_grad=True)
        Qt = torch.tensor(Q, dtype=torch.float64, requires_grad=True)
        Rt = torch.tensor(R, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            differentiable_care, (At, Bt, Qt, Rt), eps=1e-6, atol=1e-4, rtol=1e-3
        )


def test_gradcheck_differentiable_gare() -> None:
    for A, B, Q, R in _stabilizable_systems():
        gamma = 8.0
        At = torch.tensor(A, dtype=torch.float64, requires_grad=True)
        Bt = torch.tensor(B, dtype=torch.float64, requires_grad=True)
        Qt = torch.tensor(Q, dtype=torch.float64, requires_grad=True)
        Rt = torch.tensor(R, dtype=torch.float64, requires_grad=True)

        def f(a, b, q, r):  # noqa: ANN001, ANN202
            return differentiable_gare(a, b, q, r, gamma)

        assert torch.autograd.gradcheck(f, (At, Bt, Qt, Rt), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_gradcheck_differentiable_gare_with_D() -> None:
    A, B, Q, R = _stabilizable_systems()[0]
    gamma = 8.0
    D = np.array([[1.0], [0.5]])
    At = torch.tensor(A, dtype=torch.float64, requires_grad=True)
    Bt = torch.tensor(B, dtype=torch.float64, requires_grad=True)
    Qt = torch.tensor(Q, dtype=torch.float64, requires_grad=True)
    Rt = torch.tensor(R, dtype=torch.float64, requires_grad=True)
    Dt = torch.tensor(D, dtype=torch.float64, requires_grad=True)

    def f(a, b, q, r, d):  # noqa: ANN001, ANN202
        return differentiable_gare(a, b, q, r, gamma, d)

    assert torch.autograd.gradcheck(f, (At, Bt, Qt, Rt, Dt), eps=1e-6, atol=1e-4, rtol=1e-3)


# --------------------------------------------------------------------------- #
# End-to-end: scalar loss backpropagates to finite, correct grads.
# --------------------------------------------------------------------------- #
def test_care_end_to_end_backward() -> None:
    A, B, Q, R = _stabilizable_systems()[0]
    At = torch.tensor(A, dtype=torch.float64, requires_grad=True)
    Bt = torch.tensor(B, dtype=torch.float64, requires_grad=True)
    Qt = torch.tensor(Q, dtype=torch.float64, requires_grad=True)
    Rt = torch.tensor(R, dtype=torch.float64, requires_grad=True)
    P = differentiable_care(At, Bt, Qt, Rt)
    loss = (P**2).sum()
    loss.backward()
    for g in (At.grad, Bt.grad, Qt.grad, Rt.grad):
        assert g is not None
        assert torch.isfinite(g).all()
    # Cross-check dL/dQ against a finite difference on the scalar loss.
    eps = 1e-6
    Q0 = Q.copy()
    fd = np.zeros_like(Q0)
    for i in range(Q0.shape[0]):
        for j in range(Q0.shape[1]):
            Qp = Q0.copy()
            Qp[i, j] += eps
            Pp = differentiable_care(
                torch.tensor(A),
                torch.tensor(B),
                torch.tensor(Qp),
                torch.tensor(R),
            ).detach()
            Qm = Q0.copy()
            Qm[i, j] -= eps
            Pm = differentiable_care(
                torch.tensor(A),
                torch.tensor(B),
                torch.tensor(Qm),
                torch.tensor(R),
            ).detach()
            fd[i, j] = ((Pp**2).sum() - (Pm**2).sum()).item() / (2 * eps)
    assert np.allclose(Qt.grad.numpy(), fd, atol=1e-4)


def test_gare_end_to_end_backward() -> None:
    A, B, Q, R = _stabilizable_systems()[0]
    gamma = 8.0
    At = torch.tensor(A, dtype=torch.float64, requires_grad=True)
    Bt = torch.tensor(B, dtype=torch.float64, requires_grad=True)
    Qt = torch.tensor(Q, dtype=torch.float64, requires_grad=True)
    Rt = torch.tensor(R, dtype=torch.float64, requires_grad=True)
    P = differentiable_gare(At, Bt, Qt, Rt, gamma)
    loss = (P**2).sum()
    loss.backward()
    for g in (At.grad, Bt.grad, Qt.grad, Rt.grad):
        assert g is not None
        assert torch.isfinite(g).all()


def test_differentiable_care_no_grad_inputs_still_works() -> None:
    """If no input requires grad, forward still returns the correct P."""
    A, B, Q, R = _stabilizable_systems()[0]
    P = differentiable_care(
        torch.tensor(A),
        torch.tensor(B),
        torch.tensor(Q),
        torch.tensor(R),
    )
    P_ref, _ = solve_care(A, B, Q, R)
    assert torch.allclose(P.detach(), torch.tensor(P_ref), atol=1e-8)


def test_differentiable_gare_infeasible_gamma_raises() -> None:
    A, B, Q, R = _stabilizable_systems()[0]
    with pytest.raises(ValueError):
        differentiable_gare(
            torch.tensor(A, dtype=torch.float64),
            torch.tensor(B, dtype=torch.float64),
            torch.tensor(Q, dtype=torch.float64),
            torch.tensor(R, dtype=torch.float64),
            gamma=0.1,
        )
