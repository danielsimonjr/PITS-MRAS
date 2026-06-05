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

from functools import lru_cache
from typing import Optional, Tuple, Union

import numpy as np
import torch
from scipy.linalg import schur, solve_continuous_are, solve_continuous_lyapunov
from torch import Tensor


def _canonical_device_key(device: Union[str, torch.device]) -> str:
    """Canonical cache key for a device.

    Collapses equivalent specs onto one key so the ``_triu_pairs`` cache does not
    hold duplicate entries — e.g. a bare ``"cuda"`` (current device) is resolved
    to its concrete ``"cuda:<idx>"`` form when CUDA is available, matching what a
    tensor's own ``.device`` reports. Returns ``str(torch.device(...))``.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None and torch.cuda.is_available():
        dev = torch.device("cuda", torch.cuda.current_device())
    return str(dev)


@lru_cache(maxsize=128)
def _triu_pairs_cached(n: int, device_key: str) -> Tuple[Tensor, Tensor]:
    """Build the row-major upper-triangular index pair for an n x n matrix.

    Keyed on a *canonical* device string (see :func:`_canonical_device_key`).
    Bounded cache: only a handful of ``(n, device)`` combinations occur in
    practice, so a small ``maxsize`` is ample and avoids unbounded retention of
    the index tensors for the process lifetime.
    """
    idx = torch.triu_indices(n, n, device=device_key)
    return idx[0], idx[1]


def _triu_pairs(n: int, device: Union[str, torch.device]) -> Tuple[Tensor, Tensor]:
    """Cached row-major upper-triangular index pair ``(i, j)`` for an n x n matrix.

    The quadratic-basis ordering used everywhere. The returned ``(i, j)`` are
    shared, **read-only** index tensors — callers must not mutate them. Cached
    per ``(n, canonical-device)`` so the shared helpers avoid rebuilding the
    index tensors on every call (a small win on the per-step ``extract_P`` path).
    """
    return _triu_pairs_cached(n, _canonical_device_key(device))


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


def solve_gare(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    gamma: float,
    D: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Solve the H-infinity Game Algebraic Riccati Equation (GARE).

    For the system :math:`\dot x = Ax + Bu + Dw` with cost
    :math:`\int (x^\top Q x + u^\top R u - \gamma^2 \lVert w\rVert^2)\,dt`, the
    H-infinity value :math:`V = x^\top P x` uses the stabilizing solution of

    .. math::
        A^\top P + P A + Q - P\,(B R^{-1} B^\top - \gamma^{-2} D D^\top)\,P = 0.

    Solved directly via the Hamiltonian--Schur method (no iteration): the
    stabilizing :math:`P` comes from the stable invariant subspace of the
    Hamiltonian :math:`H = [[A, -M], [-Q, -A^\top]]`,
    :math:`M = B R^{-1} B^\top - \gamma^{-2} D D^\top`.

    Args:
        A, B, Q, R: system + cost matrices (as in :func:`solve_care`).
        gamma: disturbance-attenuation level. Larger ``gamma`` is easier (as
            ``gamma -> inf`` the GARE reduces to the CARE); ``gamma`` must exceed
            the H-infinity-achievable bound or no stabilizing PD solution exists.
        D: disturbance input matrix ``[n, n_w]``. Defaults to ``B`` (matched /
            input-channel disturbance).

    Returns:
        ``(P, K, L)``: the stabilizing ``P``, the robust control gain
        ``K = R^{-1} B^\top P`` (so ``u* = -K x``), and the worst-case
        disturbance gain ``L = gamma^{-2} D^\top P`` (so ``w* = L x``).

    Raises:
        ValueError: if ``gamma`` is infeasible -- the Hamiltonian has no proper
            n-dimensional stable subspace, or the resulting ``P`` is not PD, or
            the worst-case closed loop ``A - M P`` is not Hurwitz.
    """
    if D is None:
        D = B
    n = A.shape[0]
    r_inv = np.linalg.inv(R)
    M = B @ r_inv @ B.T - (1.0 / gamma**2) * D @ D.T
    ham = np.block([[A, -M], [-Q, -A.T]])
    # Ordered real Schur: eigenvalues with negative real part (the stable
    # subspace) sorted into the leading n columns.
    _, Z, sdim = schur(ham, output="real", sort="lhp")
    if sdim != n:
        raise ValueError(
            f"H-infinity GARE infeasible at gamma={gamma:.4g}: the Hamiltonian "
            f"has {sdim} stable eigenvalues, expected {n} (gamma is at or below "
            "the achievable attenuation bound)."
        )
    u11 = Z[:n, :n]
    u21 = Z[n:, :n]
    P = u21 @ np.linalg.inv(u11)
    P = 0.5 * (P + P.T)
    if np.min(np.linalg.eigvalsh(P)) <= 0:
        raise ValueError(
            f"H-infinity GARE infeasible at gamma={gamma:.4g}: solution P is not "
            "positive definite."
        )
    if not check_hurwitz(A - M @ P):
        raise ValueError(
            f"H-infinity GARE infeasible at gamma={gamma:.4g}: the worst-case "
            "closed loop A - M P is not Hurwitz."
        )
    K = r_inv @ B.T @ P
    L = (1.0 / gamma**2) * D.T @ P
    return P, K, L


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
    i, j = _triu_pairs(n, e.device)
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
    i, j = _triu_pairs(n, P.device)
    off = i != j
    return P[i, j] + torch.where(off, P[j, i], torch.zeros_like(P[i, j]))


def unpack_symmetric(vec: Tensor, n: int) -> Tensor:
    r"""Inverse of :func:`pack_symmetric`: basis coefficients -> symmetric ``[n, n]``.

    Diagonal entries take the coefficient directly; off-diagonal coefficients are
    split symmetrically across ``P[i, j]`` and ``P[j, i]``.
    """
    i, j = _triu_pairs(n, vec.device)
    off = i != j
    half = torch.where(off, vec / 2.0, vec)
    P = torch.zeros(n, n, device=vec.device, dtype=vec.dtype)
    P[i, j] = half
    P[j, i] = half
    return P
