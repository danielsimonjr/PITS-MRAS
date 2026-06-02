"""Offline IRL critic trainer (batch least-squares, §8.3).

Pre-trains the quadratic critic ``V(e) = e^T P e`` from a fixed batch of
trajectories *without* knowing the drift dynamics, using the integral
reinforcement-learning Bellman identity (Vrabie & Lewis, 2009):

    V(e(t)) - V(e(t + T)) = integral_{t}^{t+T} r(e, u) ds

Because ``V(e) = e^T P e`` is linear in the (vectorized) entries of ``P``, each
window contributes a linear equation

    (phi(e_start) - phi(e_end)) . p = integral_cost

where ``phi(e)`` is the quadratic feature map and ``p`` is the vector of
upper-triangular entries of ``P`` (using the SAME basis convention as
:meth:`QuadraticCritic.set_P`: diagonal coefficient is ``P[i, i]``, the
off-diagonal coefficient of ``e_i e_j`` is ``P[i, j] + P[j, i]``). Stacking all
windows gives an over-determined linear system solved by least squares
(Kleinman-style batch update). The trainer iterates the solve and stops when

    ||P_hat - P_opt||_F / ||P_opt||_F < tol.

Design notes (signatures NOT pinned by the source; flagged Gap G6/G7): the
function signature, the synthetic trajectory generator, and the
``(P_hat, converged, n_iters)`` return tuple are designed here. Trajectories
are rolled out from the optimal closed loop ``e_dot = (A_m - B_m K_opt) e`` with
running cost ``e^T Q e + u^T R u`` and ``u = -K_opt e`` (no external dataset).
For this consistent data the least-squares fit recovers ``P_opt`` exactly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from pits_mras.controllers.reference_models import LinearReferenceModel
    from pits_mras.models.critic import QuadraticCritic

logger = logging.getLogger(__name__)


def train_irl_critic_gd(
    critic: "QuadraticCritic",
    ref_model: "LinearReferenceModel",
    *,
    n_trajectories: int = 64,
    traj_len: int = 40,
    window_size: int = 20,
    dt: float = 0.01,
    steps: int = 300,
    lr: float = 0.1,
    seed: int = 0,
) -> list[float]:
    """Offline *gradient-descent* IRL critic fit; returns the convergence history.

    Companion to :func:`train_irl_critic` (batch least-squares, one-shot). This
    variant takes ``steps`` Adam gradient steps on the convex IRL Bellman loss
    over a FIXED batch of optimal-closed-loop windows (``u = -K_opt e``), so it
    is decoupled from control-loop stability and converges monotonically from an
    arbitrary starting ``P``. Useful for visualizing critic convergence (e.g. the
    robotic-manipulator demo's panel (d)).

    Args:
        critic: quadratic critic to fit (updated in place).
        ref_model: provides ``A_m, B_m, Q, R, K_opt, P_opt``.
        n_trajectories, traj_len, window_size, dt: synthetic-data parameters.
        steps: number of gradient steps.
        lr: Adam learning rate for the critic.
        seed: RNG seed for the synthetic data.

    Returns:
        Per-step relative Frobenius error ``||P_hat - P_opt|| / ||P_opt||``
        (length ``steps``).
    """
    from pits_mras.losses.irl import IRLBellmanLoss

    generator = torch.Generator().manual_seed(seed)
    errors, _ = _generate_optimal_trajectories(
        ref_model, n_trajectories, traj_len, dt, generator
    )
    n_windows = traj_len - window_size
    if n_windows <= 0:
        raise ValueError("traj_len must exceed window_size")
    # Stack sliding windows along the batch axis: [n_traj * n_windows, W+1, n].
    windows = [errors[:, t : t + window_size + 1, :] for t in range(n_windows)]
    e_win = torch.cat(windows, dim=0)
    u_win = -torch.einsum("ij,bwj->bwi", ref_model.K_opt, e_win)

    irl = IRLBellmanLoss(ref_model.Q, ref_model.R)
    optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    p_opt = ref_model.P_opt
    p_opt_norm = torch.linalg.norm(p_opt)

    history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = irl(critic, e_win, u_win, dt)["loss"]
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            diff = torch.linalg.norm(critic.extract_P() - p_opt)
            rel = diff / p_opt_norm if float(p_opt_norm) > 0.0 else diff
        history.append(float(rel))
    return history


def _quadratic_features(e: Tensor, state_dim: int) -> Tensor:
    """Feature map phi(e) with ``e^T P e == phi(e) . p``.

    ``p`` holds the upper-triangular coefficients in the
    :class:`QuadraticCritic` basis: the diagonal coefficient of ``e_i^2`` is
    ``P[i, i]`` and the coefficient of the cross term ``e_i e_j`` (i < j) is
    ``P[i, j] + P[j, i]``. The matching feature for that cross term is therefore
    ``e_i e_j`` (NOT doubled), so that ``(P[i,j] + P[j,i]) * e_i e_j`` reproduces
    both symmetric off-diagonal contributions.

    Args:
        e: error states, shape ``[..., state_dim]``.
        state_dim: dimension of the state.

    Returns:
        Feature tensor of shape ``[..., n_entries]`` where
        ``n_entries = state_dim * (state_dim + 1) // 2``.
    """
    feats: list[Tensor] = []
    for i in range(state_dim):
        for j in range(i, state_dim):
            feats.append(e[..., i] * e[..., j])
    return torch.stack(feats, dim=-1)


def _vec_to_P(p: Tensor, state_dim: int) -> Tensor:
    """Reconstruct the symmetric matrix P from its basis-coefficient vector.

    Inverse of the :class:`QuadraticCritic` basis: diagonal entries take the
    coefficient directly; off-diagonal coefficients are split symmetrically
    across ``P[i, j]`` and ``P[j, i]`` (matching :meth:`QuadraticCritic.set_P`).
    """
    P = torch.zeros(state_dim, state_dim, dtype=p.dtype, device=p.device)
    idx = 0
    for i in range(state_dim):
        for j in range(i, state_dim):
            if i == j:
                P[i, j] = p[idx]
            else:
                P[i, j] = p[idx] / 2.0
                P[j, i] = p[idx] / 2.0
            idx += 1
    return P


def _generate_optimal_trajectories(
    ref_model: "LinearReferenceModel",
    n_trajectories: int,
    traj_len: int,
    dt: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Roll out optimal closed-loop error trajectories and running costs.

    Returns:
        errors: ``[n_trajectories, traj_len, state_dim]`` error states.
        costs:  ``[n_trajectories, traj_len]`` instantaneous running costs
                ``r(e, u) = e^T Q e + u^T R u`` with ``u = -K_opt e``.
    """
    state_dim = ref_model.A_m.shape[0]
    A_m = ref_model.A_m
    B_m = ref_model.B_m
    K_opt = ref_model.K_opt
    Q = ref_model.Q
    R = ref_model.R
    A_cl = A_m - B_m @ K_opt  # closed-loop dynamics

    e = torch.randn(n_trajectories, state_dim, generator=generator)
    errors = torch.empty(n_trajectories, traj_len, state_dim)
    costs = torch.empty(n_trajectories, traj_len)
    for t in range(traj_len):
        errors[:, t, :] = e
        u = -torch.einsum("ij,bj->bi", K_opt, e)
        eQe = torch.einsum("bi,ij,bj->b", e, Q, e)
        uRu = torch.einsum("bi,ij,bj->b", u, R, u)
        costs[:, t] = eQe + uRu
        e_dot = torch.einsum("ij,bj->bi", A_cl, e)
        e = e + dt * e_dot
    return errors, costs


def _build_least_squares_system(
    errors: Tensor,
    costs: Tensor,
    window_size: int,
    dt: float,
    state_dim: int,
) -> tuple[Tensor, Tensor]:
    """Assemble the stacked IRL Bellman LS system ``Phi p = y``.

    For each window ``[t, t + window_size]`` the row is
    ``phi(e_t) - phi(e_{t+W})`` and the target is the trapezoidal integral of
    the running cost over the window.
    """
    _, traj_len, _ = errors.shape
    n_windows = traj_len - window_size
    if n_windows <= 0:
        raise ValueError("traj_len must exceed window_size")

    phi = _quadratic_features(errors, state_dim)  # [n_traj, traj_len, n_entries]

    # Cumulative trapezoidal integral of the running cost along time.
    cumulative = torch.zeros_like(costs)
    for t in range(1, traj_len):
        cumulative[:, t] = cumulative[:, t - 1] + 0.5 * dt * (
            costs[:, t] + costs[:, t - 1]
        )

    rows: list[Tensor] = []
    targets: list[Tensor] = []
    for t in range(n_windows):
        phi_diff = phi[:, t, :] - phi[:, t + window_size, :]  # [n_traj, n_entries]
        integral = cumulative[:, t + window_size] - cumulative[:, t]  # [n_traj]
        rows.append(phi_diff)
        targets.append(integral)
    Phi = torch.cat(rows, dim=0)  # [n_traj * n_windows, n_entries]
    y = torch.cat(targets, dim=0)  # [n_traj * n_windows]
    return Phi, y


def train_irl_critic(
    critic: "QuadraticCritic",
    ref_model: "LinearReferenceModel",
    *,
    n_trajectories: int = 64,
    traj_len: int = 40,
    window_size: int = 5,
    dt: float = 0.01,
    max_iters: int = 50,
    tol: float = 0.01,
    seed: int = 0,
) -> tuple[Tensor, bool, int]:
    """Offline batch least-squares IRL critic training (§8.3).

    Fits the critic's ``P`` to satisfy the IRL Bellman equations on a fixed
    batch of synthetic optimal-closed-loop trajectories and stops when the
    relative Frobenius error to ``P_opt`` drops below ``tol``.

    Args:
        critic: quadratic critic to fit (updated in place via ``set_P``).
        ref_model: provides ``A_m, B_m, Q, R, K_opt, P_opt``.
        n_trajectories: number of independent rollouts in the batch.
        traj_len: timesteps per rollout.
        window_size: IRL integration window length (in steps).
        dt: integration timestep.
        max_iters: maximum number of LS solves.
        tol: relative-error stopping threshold ``||P_hat - P_opt||/||P_opt||``.
        seed: RNG seed for the synthetic data.

    Returns:
        ``(P_hat, converged, n_iters)`` where ``P_hat`` is the fitted matrix,
        ``converged`` indicates the tolerance was met, and ``n_iters`` is the
        number of LS solves performed.
    """
    state_dim = ref_model.A_m.shape[0]
    generator = torch.Generator().manual_seed(seed)

    errors, costs = _generate_optimal_trajectories(
        ref_model, n_trajectories, traj_len, dt, generator
    )
    Phi, y = _build_least_squares_system(errors, costs, window_size, dt, state_dim)

    P_opt = ref_model.P_opt
    p_opt_norm = torch.linalg.norm(P_opt)

    P_hat = critic.extract_P()
    converged = False
    n_iters = 0
    for _ in range(max_iters):
        n_iters += 1
        # Batch least-squares solve of Phi p = y (the system is linear in p).
        solution = torch.linalg.lstsq(Phi, y.unsqueeze(-1))
        p = solution.solution.squeeze(-1)
        P_hat = _vec_to_P(p, state_dim)
        critic.set_P(P_hat)
        rel_err = torch.linalg.norm(P_hat - P_opt) / p_opt_norm
        logger.info("irl_trainer iter %d  rel_err=%.6g", n_iters, float(rel_err))
        if float(rel_err) < tol:
            converged = True
            break

    return P_hat, converged, n_iters
