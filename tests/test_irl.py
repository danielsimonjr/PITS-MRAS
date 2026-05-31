"""Phase-3 acceptance tests for the IRL Bellman residual loss.

Two load-bearing checks (§3.2 of the impl plan):

* ``test_irl_bellman_error_zero_at_true_value`` -- when the critic encodes the
  TRUE value function ``V(e) = eᵀ P e`` (with ``P`` the Lyapunov/CARE solution)
  along a consistent closed-loop trajectory, the IRL Bellman residual
  ``δ_IRL`` is (numerically) zero.
* ``test_irl_loss_decreases_with_correct_update`` -- starting from a wrong P̂,
  a gradient step toward the true P̂ strictly decreases ``L_IRL``.

The IRL Bellman equation is model-free: the drift matrix ``A`` never appears
in ``δ_IRL`` -- only the running cost integral and the value difference.
"""
from __future__ import annotations

import torch
from scipy.linalg import solve_continuous_lyapunov

from pits_mras.losses.irl import IRLBellmanAccumulator, IRLBellmanLoss
from pits_mras.models.critic import QuadraticCritic


def _inject_P(critic: QuadraticCritic, P: torch.Tensor) -> None:
    """Set the critic's W_c weights so that V(e) = eᵀ P e exactly.

    ``quadratic_basis`` ordering is ``[e_i e_j for j>=i]``; the diagonal
    coefficient is ``P[i,i]`` and the off-diagonal coefficient is
    ``P[i,j] + P[j,i] = 2 P[i,j]`` (symmetric P).
    """
    n = critic.state_dim
    coeffs = []
    for i in range(n):
        for j in range(i, n):
            coeffs.append(P[i, j] if i == j else P[i, j] + P[j, i])
    w = torch.stack(coeffs).to(critic.W_c.weight.dtype)
    with torch.no_grad():
        critic.W_c.weight.copy_(w.unsqueeze(0))


def _lqr_2d() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                       torch.Tensor, torch.Tensor]:
    """Return (A, B, Q, R, K, P) for a controllable double-integrator LQR.

    ``P`` is the value matrix s.t. V(e)=eᵀPe; ``K`` the optimal gain; the
    closed loop ``A_cl = A - B K`` is Hurwitz and Q_cl = Q + KᵀRK satisfies the
    Lyapunov identity ``A_clᵀP + P A_cl = -Q_cl``.
    """
    A = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
    B = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    Q = torch.eye(2, dtype=torch.float64)
    R = torch.eye(1, dtype=torch.float64)
    # A known stabilizing gain; pick K so A_cl is Hurwitz, then P from the
    # Lyapunov identity using Q_cl = Q + KᵀRK (no LQR optimality required for
    # the residual identity -- only consistency of (cost, V, trajectory)).
    K = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    A_cl = A - B @ K
    Q_cl = Q + K.transpose(-1, -2) @ R @ K
    # A_clᵀ P + P A_cl = -Q_cl  (closed-loop Lyapunov identity).
    P_np = solve_continuous_lyapunov(A_cl.T.numpy(), -Q_cl.numpy())
    P = torch.from_numpy(P_np)
    return A, B, Q, R, K, P


def _simulate(A_cl: torch.Tensor, e0: torch.Tensor, dt: float,
              steps: int) -> torch.Tensor:
    """Integrate ė = A_cl e with RK4; return trajectory [steps+1, state_dim]."""
    traj = [e0]
    e = e0
    for _ in range(steps):
        k1 = e @ A_cl.transpose(-1, -2)
        k2 = (e + 0.5 * dt * k1) @ A_cl.transpose(-1, -2)
        k3 = (e + 0.5 * dt * k2) @ A_cl.transpose(-1, -2)
        k4 = (e + dt * k3) @ A_cl.transpose(-1, -2)
        e = e + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(e)
    return torch.stack(traj, dim=0)


class TestIRLBellman:
    def test_irl_bellman_error_zero_at_true_value(self) -> None:
        A, B, Q, R, K, P = _lqr_2d()
        A_cl = A - B @ K
        dt = 1e-3
        steps = 1000  # window T = 1.0 s
        e0 = torch.tensor([[1.0, -0.5]], dtype=torch.float64)
        traj = _simulate(A_cl, e0, dt, steps)  # [steps+1, 1, state_dim]
        traj = traj.squeeze(1)                 # [steps+1, state_dim]

        u = -(traj @ K.transpose(-1, -2))      # [steps+1, action_dim]

        critic = QuadraticCritic(2).double()
        _inject_P(critic, P)  # V̂(e) = eᵀ P e exactly
        assert torch.allclose(critic.extract_P(), P, atol=1e-9)

        acc = IRLBellmanAccumulator(Q, R)
        integral = acc(traj.unsqueeze(0), u.unsqueeze(0), dt)  # [batch]

        v_end = critic(traj[-1:])    # V̂(e(t))     [1]
        v_start = critic(traj[:1])   # V̂(e(t−T))   [1]
        # Integral Bellman eq: ∫ r dτ = V(t−T) − V(t)  =>  δ = ∫ − (V_start − V_end).
        delta = integral - (v_start - v_end)
        # Residual is RK4 truncation only (~1e-4 at dt=1e-3); assert it vanishes.
        assert torch.abs(delta).max().item() < 1e-3

    def test_irl_loss_decreases_with_correct_update(self) -> None:
        A, B, Q, R, K, P = _lqr_2d()
        A_cl = A - B @ K
        dt = 1e-3
        steps = 1000
        e0 = torch.tensor([[1.0, -0.5]], dtype=torch.float64)
        traj = _simulate(A_cl, e0, dt, steps).squeeze(1)
        u = -(traj @ K.transpose(-1, -2))

        critic = QuadraticCritic(2).double()
        # Start from a WRONG P̂ (scaled down -> nonzero residual).
        _inject_P(critic, 0.3 * P)

        loss_fn = IRLBellmanLoss(Q, R)
        traj_b = traj.unsqueeze(0)
        u_b = u.unsqueeze(0)

        out0 = loss_fn(critic, traj_b, u_b, dt)
        loss0 = out0["loss"]
        assert loss0.item() > 0.0

        opt = torch.optim.SGD(critic.parameters(), lr=1e-3)
        opt.zero_grad()
        loss0.backward()
        opt.step()

        out1 = loss_fn(critic, traj_b, u_b, dt)
        assert out1["loss"].item() < loss0.item()
