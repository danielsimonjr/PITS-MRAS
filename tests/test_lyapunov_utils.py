"""Phase-1 unit tests for ``pits_mras.utils.lyapunov`` (IP §4.3).

Covers ``solve_lyapunov``, ``solve_care``, ``check_hurwitz`` and
``lyapunov_derivative`` with numerical assertions derived from the math. The
``kleinman``/``quadratic_basis`` tests live in
``test_identity_lyapunov_value.py`` (the reserved Identity-1 file).
"""

import numpy as np
import pytest
import torch

from pits_mras.utils.lyapunov import (
    check_hurwitz,
    lyapunov_derivative,
    pack_symmetric,
    quadratic_basis,
    solve_care,
    solve_lyapunov,
    unpack_symmetric,
)


def test_solve_lyapunov_identity_gate() -> None:
    """A=-I, Q=I  =>  P = 0.5 I  (the §13 acceptance gate)."""
    A = -np.eye(2)
    Q = np.eye(2)
    P = solve_lyapunov(A, Q)
    assert np.allclose(P, 0.5 * np.eye(2), atol=1e-12)


def test_solve_lyapunov_satisfies_equation() -> None:
    """The returned P satisfies A_mᵀP + PA_m = -Q."""
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    Q = np.array([[2.0, 0.0], [0.0, 1.0]])
    P = solve_lyapunov(A, Q)
    residual = A.T @ P + P @ A + Q
    assert np.allclose(residual, np.zeros((2, 2)), atol=1e-10)
    # P must be symmetric positive definite.
    assert np.allclose(P, P.T, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(P)) > 0


def test_solve_lyapunov_raises_on_non_hurwitz() -> None:
    """Non-Hurwitz A_m yields a non-PD P -> ValueError."""
    A = np.array([[1.0, 0.0], [0.0, 1.0]])  # unstable
    Q = np.eye(2)
    with pytest.raises(ValueError):
        solve_lyapunov(A, Q)


def test_solve_care_satisfies_are() -> None:
    """P, K from solve_care satisfy the CARE and K = R⁻¹BᵀP."""
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    Q = np.eye(2)
    R = np.array([[1.0]])
    P, K = solve_care(A, B, Q, R)
    are = A.T @ P + P @ A - P @ B @ np.linalg.solve(R, B.T @ P) + Q
    assert np.allclose(are, np.zeros((2, 2)), atol=1e-8)
    assert np.allclose(K, np.linalg.solve(R, B.T @ P), atol=1e-10)


def test_check_hurwitz_true_and_false() -> None:
    """check_hurwitz: True for stable, False for unstable A."""
    stable = np.array([[-1.0, 0.0], [0.0, -2.0]])
    unstable = np.array([[1.0, 0.0], [0.0, -2.0]])
    marginal = np.array([[0.0, 1.0], [-1.0, 0.0]])  # eigenvalues ±i
    assert check_hurwitz(stable) is True
    assert check_hurwitz(unstable) is False
    assert check_hurwitz(marginal) is False


def test_lyapunov_derivative_matches_formula() -> None:
    """V̇ = 2eᵀP(A_m e + B u), batched."""
    e = torch.tensor([[1.0, 2.0], [0.5, -1.0]], dtype=torch.float64)
    P = torch.eye(2, dtype=torch.float64)
    A_m = torch.tensor([[0.0, 1.0], [-1.0, -1.0]], dtype=torch.float64)
    B = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    u = torch.tensor([[0.3], [-0.2]], dtype=torch.float64)

    v_dot = lyapunov_derivative(e, P, A_m, B, u)

    # Reference computation row by row.
    expected = []
    for k in range(2):
        ek = e[k]
        edot = A_m @ ek + B @ u[k]
        expected.append(2.0 * (ek @ (P @ edot)).item())
    expected_t = torch.tensor(expected, dtype=torch.float64)
    assert v_dot.shape == (2,)
    assert torch.allclose(v_dot, expected_t, atol=1e-10)


def test_pack_unpack_symmetric_round_trip() -> None:
    """unpack_symmetric(pack_symmetric(P)) recovers a symmetric P."""
    torch.manual_seed(0)
    for n in (1, 2, 4):
        a = torch.randn(n, n)
        P = a + a.T  # symmetric
        vec = pack_symmetric(P)
        assert vec.shape == (n * (n + 1) // 2,)
        P_back = unpack_symmetric(vec, n)
        assert torch.allclose(P_back, P, atol=1e-6)
        assert torch.allclose(P_back, P_back.T, atol=1e-6)


def test_triu_pairs_canonicalizes_equivalent_devices_and_bounds_cache() -> None:
    """_triu_pairs dedups equivalent device specs (str vs torch.device) onto a
    single cache entry, and the cache is bounded (not maxsize=None)."""
    from pits_mras.utils import lyapunov as lyap

    lyap._triu_pairs_cached.cache_clear()
    a = lyap._triu_pairs(3, "cpu")
    b = lyap._triu_pairs(3, torch.device("cpu"))
    # Same canonical key -> the identical cached tensors (one entry, not two).
    assert a[0] is b[0] and a[1] is b[1]
    assert lyap._triu_pairs_cached.cache_info().currsize == 1
    # Cache is bounded.
    assert lyap._triu_pairs_cached.cache_info().maxsize is not None


def test_canonical_device_key_cpu() -> None:
    """The device-key canonicalizer accepts both str and torch.device for CPU."""
    from pits_mras.utils.lyapunov import _canonical_device_key

    assert _canonical_device_key("cpu") == "cpu"
    assert _canonical_device_key(torch.device("cpu")) == "cpu"


def test_quadratic_form_equals_basis_dot_packed() -> None:
    """e^T P e == quadratic_basis(e) . pack_symmetric(P) (the shared convention)."""
    torch.manual_seed(1)
    n = 3
    a = torch.randn(n, n)
    P = a + a.T
    e = torch.randn(5, n)
    quad = torch.einsum("bi,ij,bj->b", e, P, e)
    via_basis = quadratic_basis(e) @ pack_symmetric(P)
    assert torch.allclose(quad, via_basis, atol=1e-5)
