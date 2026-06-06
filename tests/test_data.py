"""Tests for the reusable trajectory data package (Gap G7).

Covers:
  * generate_synthetic_trajectories — shapes, seeded reproducibility, and a
    hand-checked forward-Euler rollout,
  * TrajectoryDataset — construction + shape validation, windowed __getitem__
    round-trip,
  * make_dataloader — batch shapes and batch_size handling,
  * opt-in pretrain integration — default path unchanged + dataset path runs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pits_mras.config import NetworkConfig, PhysicsConfig, PITSMRASConfig
from pits_mras.data import (
    TrajectoryDataset,
    generate_synthetic_trajectories,
    make_dataloader,
)
from pits_mras.models import PITNN
from pits_mras.training.pretrain import pretrain_pitnn


# --------------------------------------------------------------------------- #
# Fixtures.
# --------------------------------------------------------------------------- #
def _ab() -> tuple[torch.Tensor, torch.Tensor]:
    A_m = torch.tensor([[0.0, 1.0], [-1.0, -1.0]])
    B_m = torch.tensor([[0.0], [1.0]])
    return A_m, B_m


def _small_cfg() -> PITSMRASConfig:
    cfg = PITSMRASConfig()
    cfg.network = NetworkConfig(
        input_dim=2,
        hidden_dim=16,
        output_dim=2,
        lstm_layers=1,
        attention_heads=2,
        embedding_dim=8,
    )
    cfg.physics = PhysicsConfig(
        n_generalized_coords=1,
        hamiltonian_hidden=16,
        dissipation_hidden=8,
    )
    return cfg


# --------------------------------------------------------------------------- #
# generate_synthetic_trajectories.
# --------------------------------------------------------------------------- #
def test_generate_shapes() -> None:
    A_m, B_m = _ab()
    states, controls = generate_synthetic_trajectories(
        A_m, B_m, dt=0.01, n_trajectories=5, n_steps=20, control_dim=1, seed=0
    )
    assert states.shape == (5, 20, 2)
    assert controls.shape == (5, 20, 1)
    assert states.dtype == torch.float32
    assert controls.dtype == torch.float32
    assert torch.isfinite(states).all()


def test_generate_seed_reproducible_and_distinct() -> None:
    A_m, B_m = _ab()
    s1, c1 = generate_synthetic_trajectories(
        A_m, B_m, dt=0.01, n_trajectories=3, n_steps=10, control_dim=1, seed=42
    )
    s2, c2 = generate_synthetic_trajectories(
        A_m, B_m, dt=0.01, n_trajectories=3, n_steps=10, control_dim=1, seed=42
    )
    assert torch.equal(s1, s2)
    assert torch.equal(c1, c2)
    s3, _ = generate_synthetic_trajectories(
        A_m, B_m, dt=0.01, n_trajectories=3, n_steps=10, control_dim=1, seed=7
    )
    assert not torch.equal(s1, s3)


def test_generate_matches_forward_euler() -> None:
    """Hand-check the rollout against ``x <- x + dt * (A x + B u)``."""
    A_m, B_m = _ab()
    dt = 0.05
    states, controls = generate_synthetic_trajectories(
        A_m, B_m, dt=dt, n_trajectories=2, n_steps=6, control_dim=1, seed=1
    )
    for t in range(states.shape[1] - 1):
        x = states[:, t, :]
        u = controls[:, t, :]
        x_dot = torch.einsum("ij,bj->bi", A_m, x) + torch.einsum("ij,bj->bi", B_m, u)
        expected = x + dt * x_dot
        torch.testing.assert_close(states[:, t + 1, :], expected, rtol=1e-5, atol=1e-6)


def test_generate_accepts_numpy() -> None:
    A_m = np.array([[0.0, 1.0], [-1.0, -1.0]])
    B_m = np.array([[0.0], [1.0]])
    states, controls = generate_synthetic_trajectories(
        A_m, B_m, dt=0.01, n_trajectories=2, n_steps=5, control_dim=1, seed=0
    )
    assert states.shape == (2, 5, 2)


def test_generate_bad_shapes_raise() -> None:
    A_m, B_m = _ab()
    with pytest.raises(ValueError):
        generate_synthetic_trajectories(
            A_m, B_m, dt=0.01, n_trajectories=1, n_steps=5, control_dim=2, seed=0
        )  # control_dim != B_m.shape[1]
    with pytest.raises(ValueError):
        generate_synthetic_trajectories(
            torch.zeros(2, 3), B_m, dt=0.01, n_trajectories=1, n_steps=5, control_dim=1
        )  # A_m not square
    with pytest.raises(ValueError):
        generate_synthetic_trajectories(
            A_m, B_m, dt=-0.01, n_trajectories=1, n_steps=5, control_dim=1
        )  # dt <= 0


# --------------------------------------------------------------------------- #
# TrajectoryDataset.
# --------------------------------------------------------------------------- #
def test_dataset_len_and_window_shapes() -> None:
    A_m, B_m = _ab()
    states, controls = generate_synthetic_trajectories(
        A_m, B_m, dt=0.01, n_trajectories=4, n_steps=12, control_dim=1, seed=0
    )
    W = 3
    ds = TrajectoryDataset(states, controls, memory_horizon=W)
    # windows per traj = T - 1 - W = 12 - 1 - 3 = 8; total = 4 * 8 = 32.
    assert len(ds) == 4 * (12 - 1 - W)
    sample = ds[0]
    assert sample["state_hist"].shape == (W, 2)
    assert sample["control_hist"].shape == (W, 1)
    assert sample["state"].shape == (2,)
    assert sample["control"].shape == (1,)
    assert sample["next_state"].shape == (2,)


def test_dataset_window_roundtrip_values() -> None:
    """Windows index the underlying trajectories exactly."""
    # Single deterministic trajectory: state[t] = [t, 10+t]; control[t] = [t].
    T = 8
    states = torch.stack([torch.arange(T).float(), 10 + torch.arange(T).float()], dim=1)
    controls = torch.arange(T).float().unsqueeze(1)
    W = 2
    ds = TrajectoryDataset(states, controls, memory_horizon=W)
    # First valid current index i = W = 2 -> sample 0.
    s0 = ds[0]
    torch.testing.assert_close(s0["state_hist"], states[0:2])
    torch.testing.assert_close(s0["control_hist"], controls[0:2])
    torch.testing.assert_close(s0["state"], states[2])
    torch.testing.assert_close(s0["control"], controls[2])
    torch.testing.assert_close(s0["next_state"], states[3])


def test_dataset_single_trajectory_promoted() -> None:
    """A single [T, dim] trajectory is treated as one trajectory."""
    states = torch.randn(10, 2)
    controls = torch.randn(10, 1)
    ds = TrajectoryDataset(states, controls, memory_horizon=3)
    assert ds.n_traj == 1
    assert len(ds) == 10 - 1 - 3


def test_dataset_list_of_trajectories() -> None:
    trajs_s = [torch.randn(9, 2), torch.randn(9, 2)]
    trajs_c = [torch.randn(9, 1), torch.randn(9, 1)]
    ds = TrajectoryDataset(trajs_s, trajs_c, memory_horizon=2)
    assert ds.n_traj == 2
    assert len(ds) == 2 * (9 - 1 - 2)


def test_dataset_bad_shapes_raise() -> None:
    with pytest.raises(ValueError):
        TrajectoryDataset(torch.randn(10, 2), torch.randn(8, 1), memory_horizon=2)  # T mismatch
    with pytest.raises(ValueError):
        TrajectoryDataset(torch.randn(3, 2), torch.randn(3, 1), memory_horizon=5)  # too short
    with pytest.raises(ValueError):
        TrajectoryDataset(torch.randn(10, 2), torch.randn(10, 1), memory_horizon=0)  # W < 1
    with pytest.raises(ValueError):
        TrajectoryDataset(torch.randn(10), torch.randn(10, 1), memory_horizon=2)  # 1D state


# --------------------------------------------------------------------------- #
# make_dataloader.
# --------------------------------------------------------------------------- #
def test_dataloader_batch_shapes_and_size() -> None:
    A_m, B_m = _ab()
    states, controls = generate_synthetic_trajectories(
        A_m, B_m, dt=0.01, n_trajectories=4, n_steps=12, control_dim=1, seed=0
    )
    W = 3
    ds = TrajectoryDataset(states, controls, memory_horizon=W)
    loader = make_dataloader(ds, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    assert batch["state_hist"].shape == (8, W, 2)
    assert batch["control_hist"].shape == (8, W, 1)
    assert batch["state"].shape == (8, 2)
    assert batch["control"].shape == (8, 1)
    assert batch["next_state"].shape == (8, 2)


def test_dataloader_batch_size_respected() -> None:
    A_m, B_m = _ab()
    states, controls = generate_synthetic_trajectories(
        A_m, B_m, dt=0.01, n_trajectories=2, n_steps=12, control_dim=1, seed=0
    )
    ds = TrajectoryDataset(states, controls, memory_horizon=3)
    loader = make_dataloader(ds, batch_size=4, shuffle=False, drop_last=True)
    for batch in loader:
        assert batch["state"].shape[0] == 4


def test_dataloader_bad_batch_size_raises() -> None:
    A_m, B_m = _ab()
    states, controls = generate_synthetic_trajectories(
        A_m, B_m, dt=0.01, n_trajectories=2, n_steps=12, control_dim=1, seed=0
    )
    ds = TrajectoryDataset(states, controls, memory_horizon=3)
    with pytest.raises(ValueError):
        make_dataloader(ds, batch_size=0)


# --------------------------------------------------------------------------- #
# Opt-in pretrain integration (Gap G7).
# --------------------------------------------------------------------------- #
def test_pretrain_dataset_none_unchanged() -> None:
    """Default path (dataset=None) is byte-for-byte the prior behaviour.

    Run twice with the same seed and confirm the loss series match, then confirm
    explicitly passing dataset=None equals omitting it.
    """
    cfg = _small_cfg()
    torch.manual_seed(0)
    p1 = PITNN(cfg.network, cfg.physics)
    h1 = pretrain_pitnn(p1, cfg, epochs=3, batch_size=8, seed=1)
    torch.manual_seed(0)
    p2 = PITNN(cfg.network, cfg.physics)
    h2 = pretrain_pitnn(p2, cfg, epochs=3, batch_size=8, seed=1, dataset=None)
    torch.testing.assert_close(
        torch.tensor(h1["total_loss"]), torch.tensor(h2["total_loss"]), rtol=1e-6, atol=1e-7
    )


def test_pretrain_with_dataset_runs_finite() -> None:
    """Pre-training driven by a provided dataset runs and stays finite."""
    cfg = _small_cfg()
    A_m, B_m = _ab()
    states, controls = generate_synthetic_trajectories(
        A_m, B_m, dt=0.01, n_trajectories=6, n_steps=20, control_dim=1, seed=0
    )
    ds = TrajectoryDataset(states, controls, memory_horizon=4)
    pitnn = PITNN(cfg.network, cfg.physics)
    history = pretrain_pitnn(pitnn, cfg, epochs=4, batch_size=8, seed=1, dataset=ds)
    assert len(history["total_loss"]) == 4
    assert all(math.isfinite(v) for v in history["total_loss"])
    assert all(math.isfinite(v) for v in history["data_loss"])
