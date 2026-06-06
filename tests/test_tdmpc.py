"""TD-MPC2 module tests (Connection 9 — learned-model planning).

Covers :class:`pits_mras.models.tdmpc.WorldModel`,
:class:`pits_mras.models.tdmpc.MPPIPlanner`, and
:func:`pits_mras.training.tdmpc.tdmpc_update`.

Structural checks pin shapes / finiteness / differentiability / a single
optimization step. The crux is the FALSIFIABLE PLANNER TEST: with the
WorldModel components overridden to be the GROUND TRUTH of a known
linear-quadratic problem (identity encoder, ``next(z,a)=Az+Ba``,
``reward=-(zQz+aRa)``, terminal ``value=-zP*z`` with ``P*`` from
:func:`solve_care`), the MPPI planner's first action from a nonzero state must
recover the LQR-optimal ``a* = -K* z0``. This proves the planner genuinely
optimizes over the learned model. The assertion is a loose relative error
``||a_planned - a*|| / ||a*|| < 0.25`` with a fixed seed and enough
samples/iterations to stay non-flaky.
"""

import numpy as np
import torch

from pits_mras.models.tdmpc import MPPIPlanner, WorldModel
from pits_mras.training.tdmpc import tdmpc_update
from pits_mras.utils.lyapunov import solve_care


# --------------------------------------------------------------------------- #
# WorldModel structural checks.
# --------------------------------------------------------------------------- #
def test_worldmodel_encode_shape() -> None:
    """encode maps [B, state_dim] -> [B, latent_dim]."""
    wm = WorldModel(state_dim=4, action_dim=2, latent_dim=8, hidden=(16, 16))
    s = torch.randn(5, 4)
    z = wm.encode(s)
    assert z.shape == (5, 8)
    assert torch.isfinite(z).all()


def test_worldmodel_next_shape() -> None:
    """next maps (z, a) -> next latent of shape [B, latent_dim]."""
    wm = WorldModel(state_dim=4, action_dim=2, latent_dim=8, hidden=(16, 16))
    z = torch.randn(5, 8)
    a = torch.randn(5, 2)
    z_next = wm.next(z, a)
    assert z_next.shape == (5, 8)
    assert torch.isfinite(z_next).all()


def test_worldmodel_reward_and_value_shapes() -> None:
    """reward(z,a) and value(z) and Q(z,a) are each [B, 1]."""
    wm = WorldModel(state_dim=4, action_dim=2, latent_dim=8, hidden=(16, 16))
    z = torch.randn(5, 8)
    a = torch.randn(5, 2)
    assert wm.reward(z, a).shape == (5, 1)
    assert wm.value(z).shape == (5, 1)
    assert wm.Q(z, a).shape == (5, 1)


def test_worldmodel_is_differentiable() -> None:
    """Gradients flow through the full encode->next->reward chain."""
    wm = WorldModel(state_dim=3, action_dim=2, latent_dim=6, hidden=(16, 16))
    s = torch.randn(4, 3)
    a = torch.randn(4, 2)
    z = wm.encode(s)
    z_next = wm.next(z, a)
    loss = wm.reward(z, a).mean() + wm.value(z_next).mean()
    loss.backward()
    grads = [p.grad for p in wm.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum().item() > 0.0 for g in grads)


# --------------------------------------------------------------------------- #
# tdmpc_update structural checks.
# --------------------------------------------------------------------------- #
def _synthetic_batch(state_dim: int, action_dim: int, B: int = 32) -> dict[str, torch.Tensor]:
    return {
        "state": torch.randn(B, state_dim),
        "action": torch.tanh(torch.randn(B, action_dim)),
        "reward": torch.randn(B, 1),
        "next_state": torch.randn(B, state_dim),
        "done": torch.zeros(B, 1),
    }


def test_tdmpc_update_returns_finite_losses() -> None:
    """One update produces finite consistency / reward / value losses."""
    wm = WorldModel(state_dim=3, action_dim=2, latent_dim=6, hidden=(16, 16))
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    out = tdmpc_update(wm, _synthetic_batch(3, 2), opt, gamma=0.99)
    for key in ("consistency_loss", "reward_loss", "value_loss", "total_loss"):
        assert key in out
        assert torch.isfinite(torch.tensor(out[key]))


def test_tdmpc_update_changes_params() -> None:
    """A single update step changes the world-model parameters."""
    wm = WorldModel(state_dim=3, action_dim=2, latent_dim=6, hidden=(16, 16))
    opt = torch.optim.Adam(wm.parameters(), lr=1e-2)
    before = [p.detach().clone() for p in wm.parameters()]
    tdmpc_update(wm, _synthetic_batch(3, 2), opt, gamma=0.99)
    changed = any(not torch.allclose(b, a) for b, a in zip(before, wm.parameters()))
    assert changed


# --------------------------------------------------------------------------- #
# Ground-truth LQR world model for the falsifiable planner test.
# --------------------------------------------------------------------------- #
def _lqr_world_model() -> tuple[WorldModel, np.ndarray, np.ndarray]:
    r"""Return a WorldModel whose components are the GROUND TRUTH of an LQR.

    Discretizes a small continuous LQR ``(A_c, B_c, Q, R)`` with timestep ``dt``
    via the matrix exponential, so that one planner step matches the optimal
    continuous controller in the small-``dt`` limit. The WorldModel methods are
    overridden:

    * ``encode`` = identity (latent == state),
    * ``next(z, a) = z + dt * (A_c z + B_c a)`` (Euler discrete dynamics),
    * ``reward(z, a) = -dt * (z^T Q z + a^T R a)`` (running cost),
    * ``value(z) = -z^T P* z`` (continuous CARE value),

    and returns ``(wm, K_star, _)`` with ``a* = -K_star z`` the LQR-optimal
    action.
    """
    A_c = np.array([[0.0, 1.0], [-1.0, -0.5]], dtype=np.float64)
    B_c = np.array([[0.0], [1.0]], dtype=np.float64)
    Q = np.eye(2, dtype=np.float64)
    R = np.array([[1.0]], dtype=np.float64)
    P_star, K_star = solve_care(A_c, B_c, Q, R)
    dt = 0.05

    wm = WorldModel(state_dim=2, action_dim=1, latent_dim=2, hidden=(8,))

    At = torch.tensor(A_c, dtype=torch.float32)
    Bt = torch.tensor(B_c, dtype=torch.float32)
    Qt = torch.tensor(Q, dtype=torch.float32)
    Rt = torch.tensor(R, dtype=torch.float32)
    Pt = torch.tensor(P_star, dtype=torch.float32)

    def encode(s: torch.Tensor) -> torch.Tensor:
        return s

    def next_(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return z + dt * (z @ At.T + a @ Bt.T)

    def reward(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        zq = (z @ Qt * z).sum(dim=-1, keepdim=True)
        ar = (a @ Rt * a).sum(dim=-1, keepdim=True)
        return -dt * (zq + ar)

    def value(z: torch.Tensor) -> torch.Tensor:
        return -(z @ Pt * z).sum(dim=-1, keepdim=True)

    wm.encode = encode  # type: ignore[method-assign]
    wm.next = next_  # type: ignore[method-assign]
    wm.reward = reward  # type: ignore[method-assign]
    wm.value = value  # type: ignore[method-assign]
    return wm, K_star, P_star


# --------------------------------------------------------------------------- #
# Falsifiable planner test: MPPI recovers the LQR-optimal action.
# --------------------------------------------------------------------------- #
def test_mppi_recovers_lqr_optimal_action() -> None:
    """MPPI plan on the ground-truth LQR model matches a* = -K* z0.

    Loose relative error ``||a_planned - a*|| / ||a*|| < 0.25`` with a fixed
    seed and a generous sample/iteration budget so the result is non-flaky.
    """
    wm, K_star, _ = _lqr_world_model()
    z0 = torch.tensor([[1.0, -0.5]], dtype=torch.float32)
    a_star = -(z0.numpy() @ K_star.T)  # [1, action_dim]

    planner = MPPIPlanner(
        action_dim=1,
        horizon=30,
        num_samples=512,
        iterations=8,
        num_elites=64,
        temperature=0.5,
        gamma=1.0,
        action_low=-10.0,
        action_high=10.0,
        init_std=2.0,
    )
    gen = torch.Generator().manual_seed(0)
    a_planned = planner.plan(wm, z0, generator=gen)

    assert a_planned.shape == (1, 1)
    rel_err = float(np.linalg.norm(a_planned.numpy() - a_star) / np.linalg.norm(a_star))
    assert rel_err < 0.25, f"MPPI did not recover LQR optimum: rel_err={rel_err:.3f}"


def test_mppi_more_samples_not_worse() -> None:
    """Increasing the sample budget does not degrade the planned-action error."""
    wm, K_star, _ = _lqr_world_model()
    z0 = torch.tensor([[1.0, -0.5]], dtype=torch.float32)
    a_star = -(z0.numpy() @ K_star.T)

    def _err(num_samples: int) -> float:
        planner = MPPIPlanner(
            action_dim=1,
            horizon=30,
            num_samples=num_samples,
            iterations=8,
            num_elites=max(8, num_samples // 8),
            temperature=0.5,
            gamma=1.0,
            action_low=-10.0,
            action_high=10.0,
            init_std=2.0,
        )
        gen = torch.Generator().manual_seed(0)
        a = planner.plan(wm, z0, generator=gen)
        return float(np.linalg.norm(a.numpy() - a_star) / np.linalg.norm(a_star))

    err_small = _err(128)
    err_large = _err(512)
    # More samples should not be meaningfully worse (small slack for stochasticity).
    assert err_large <= err_small + 0.1


def test_mppi_action_within_bounds() -> None:
    """The planned action respects the configured action bounds."""
    wm, _, _ = _lqr_world_model()
    z0 = torch.tensor([[3.0, 2.0]], dtype=torch.float32)
    low, high = -1.0, 1.0
    planner = MPPIPlanner(
        action_dim=1,
        horizon=20,
        num_samples=256,
        iterations=5,
        num_elites=32,
        temperature=0.5,
        gamma=1.0,
        action_low=low,
        action_high=high,
        init_std=1.0,
    )
    gen = torch.Generator().manual_seed(1)
    a = planner.plan(wm, z0, generator=gen)
    assert a.min().item() >= low - 1e-5
    assert a.max().item() <= high + 1e-5
