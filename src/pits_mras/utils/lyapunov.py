r"""Lyapunov / Riccati engine (IP §4.3). "The mathematical engine for all P."

Owning phase: Phase 1 (Foundation Layer).

Identity 1 foundation. Six scipy-backed functions (IP §4.3):
``solve_lyapunov``, ``kleinman_iteration``, ``solve_care``, ``check_hurwitz``,
``lyapunov_derivative`` (:math:`\dot V = 2e^\top P(A_m e+Bu)`), ``quadratic_basis``.

Key identities (see §3.2):
- ``A_mᵀP + PA_m = −Q`` is both the MRAS Lyapunov equation AND Kleinman's
  policy-evaluation step for the tracking-error LQR.
- ``scipy.linalg.solve_continuous_lyapunov`` solves ``AᵀP + PA = −Q``.
- ``scipy.linalg.solve_continuous_are`` solves the full CARE for policy
  improvement.

Phase-1 sanity gate (IP §13): ``solve_lyapunov(-I, I)`` must return
``[[0.5, 0], [0, 0.5]]``.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov
from torch import Tensor


def solve_lyapunov(A_m: np.ndarray, Q: np.ndarray) -> np.ndarray:
    r"""Solve :math:`A_m^\top P + P A_m = -Q` for :math:`P \succ 0`.

    ``A_m`` must be Hurwitz (all eigenvalues have negative real parts).
    Returns ``P`` as a numpy array with the same shape as ``Q``.

    Raises:
        ValueError: if the resulting ``P`` is not positive definite (which
            indicates ``A_m`` is not Hurwitz).
    """
    P = solve_continuous_lyapunov(A_m.T, -Q)
    eigvals = np.linalg.eigvalsh(P)
    if np.min(eigvals) <= 0:
        raise ValueError(
            f"P is not positive definite (min eigenvalue = {np.min(eigvals):.4e}). "
            "Check that A_m is Hurwitz."
        )
    return P


def kleinman_iteration(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    K_init: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Kleinman's policy iteration (1968) for the CARE.

    Alternates between:

    - Step 1 (policy evaluation): solve
      :math:`(A-BK)^\top P + P(A-BK) + Q + K^\top R K = 0`.
    - Step 2 (policy improvement): :math:`K \leftarrow R^{-1} B^\top P`.

    Returns ``(P_star, K_star)`` at convergence.

    Raises:
        RuntimeError: if the closed-loop Lyapunov solve fails (non-Hurwitz
            ``A - BK``) or the iteration does not converge within ``max_iter``.
    """
    n = A.shape[0]
    K = K_init if K_init is not None else np.zeros((R.shape[0], n))
    R_inv = np.linalg.inv(R)
    delta = np.inf
    for i in range(max_iter):
        A_cl = A - B @ K
        try:
            P = solve_continuous_lyapunov(A_cl.T, -(Q + K.T @ R @ K))
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"Lyapunov solve failed at iteration {i}. "
                "Closed-loop A_cl may not be Hurwitz — check initial K."
            ) from exc
        K_new = R_inv @ B.T @ P
        delta = float(np.linalg.norm(K_new - K, ord="fro"))
        K = K_new
        if delta < tol:
            return P, K
    raise RuntimeError(
        f"Kleinman iteration did not converge in {max_iter} steps "
        f"(final Δ‖K‖_F = {delta:.4e})."
    )


def solve_care(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Solve the Continuous Algebraic Riccati Equation directly via scipy.

    Returns ``(P_star, K_star)`` where :math:`K_\star = R^{-1} B^\top P_\star`.
    """
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return P, K


def check_hurwitz(A: np.ndarray, tol: float = 1e-6) -> bool:
    """Return ``True`` if all eigenvalues of ``A`` have negative real parts."""
    return bool(np.all(np.real(np.linalg.eigvals(A)) < -tol))


def lyapunov_derivative(
    e: Tensor, P: Tensor, A_m: Tensor, B: Tensor, u: Tensor
) -> Tensor:
    r"""Compute :math:`\dot V = 2 e^\top P (A_m e + B u)` analytically.

    For a purely linear error dynamics :math:`\dot e = A_m e + B u`, this is
    exact.

    Shapes: ``e`` [batch, n], ``P`` [n, n], ``A_m`` [n, n], ``B`` [n, m],
    ``u`` [batch, m]. Returns ``V_dot`` of shape [batch].
    """
    e_dot_approx = e @ A_m.T + u @ B.T  # [batch, n]
    Pe = e @ P  # [batch, n]
    V_dot = 2.0 * (Pe * e_dot_approx).sum(dim=-1)  # [batch]
    return V_dot


def quadratic_basis(e: Tensor) -> Tensor:
    r"""Upper-triangular Kronecker product basis for quadratic forms.

    For a linear critic :math:`\hat V(e) = W^\top \phi(e)`, this basis gives
    :math:`\hat V = e^\top \hat P e` exactly. For ``e`` of shape [batch, n],
    returns :math:`\phi(e)` of shape [batch, n*(n+1)//2].

    Ordering: ``[e1², e1·e2, e1·e3, ..., e2², e2·e3, ..., en²]`` (the row-major
    upper-triangular order of :func:`torch.triu_indices`). Indexing is on the
    last axis, so leading batch/time dims are supported: ``[..., n]`` ->
    ``[..., n*(n+1)//2]``.
    """
    n = e.shape[-1]
    i, j = torch.triu_indices(n, n, device=e.device)
    return e[..., i] * e[..., j]  # [..., n*(n+1)//2]


def pack_symmetric(P: Tensor) -> Tensor:
    r"""Pack a symmetric ``[n, n]`` matrix into its quadratic-basis coefficients.

    The single source of truth for the basis convention shared by
    :func:`quadratic_basis`, :class:`~pits_mras.models.critic.QuadraticCritic`
    (``W_c``) and the IRL trainer: the diagonal coefficient of :math:`e_i^2` is
    ``P[i, i]`` and the coefficient of the cross term :math:`e_i e_j` (``i < j``)
    is ``P[i, j] + P[j, i]``, so that
    :math:`e^\top P e = \phi(e)^\top \mathrm{pack}(P)`.

    Returns a vector of shape ``[n*(n+1)//2]`` in the same row-major
    upper-triangular order as :func:`quadratic_basis`.
    """
    n = P.shape[-1]
    i, j = torch.triu_indices(n, n, device=P.device)
    off = i != j
    return P[i, j] + torch.where(off, P[j, i], torch.zeros_like(P[i, j]))


def unpack_symmetric(vec: Tensor, n: int) -> Tensor:
    r"""Inverse of :func:`pack_symmetric`: basis coefficients -> symmetric ``[n, n]``.

    Diagonal entries take the coefficient directly; off-diagonal coefficients are
    split symmetrically across ``P[i, j]`` and ``P[j, i]``.
    """
    i, j = torch.triu_indices(n, n, device=vec.device)
    off = i != j
    half = torch.where(off, vec / 2.0, vec)
    P = torch.zeros(n, n, device=vec.device, dtype=vec.dtype)
    P[i, j] = half
    P[j, i] = half
    return P
