"""Reusable trajectory dataset + synthetic generator + loader (Gap G7).

Until now the training pipelines (``training/pretrain.py``,
``training/cotrain.py``) synthesized their data *inline* -- there was no
on-disk-or-in-memory dataset format, no reusable generator, and no loader (Gap
G7). This module factors out the smallest reusable surface that those pipelines
need:

* :func:`generate_synthetic_trajectories` -- a seedable forward-Euler rollout of
  the SAME synthetic plant the inline co-training code uses
  (``x_dot = A_m x + B_m u``; cf. ``training.cotrain._synthetic_plant_step``),
  producing batched trajectory tensors.
* :class:`TrajectoryDataset` -- a :class:`torch.utils.data.Dataset` that holds
  one or more ``(state, control)`` trajectories and yields the *windowed*
  samples the PITNN consumes: a history window of ``memory_horizon`` past steps
  (state + control) plus the current step's state/control and the next-state
  target.
* :func:`make_dataloader` -- a thin :class:`torch.utils.data.DataLoader`
  convenience wrapper with the default collate (which stacks the per-sample
  dicts into batched dicts).

This package is **additive and opt-in**: nothing in the existing training path
imports it unless a caller explicitly passes a dataset/loader in. The default
training behaviour (inline synthetic data, identical RNG consumption) is
unchanged.

Shape conventions (float32 throughout):

* A single trajectory's states are ``[T, state_dim]`` and controls
  ``[T, control_dim]``; a *batch* of trajectories is ``[n_traj, T, dim]``.
* A windowed sample (one ``__getitem__``) yields, for window length ``W ==
  memory_horizon``:

  - ``state_hist``   ``[W, state_dim]``   past states  ``x[i-W : i]``
  - ``control_hist`` ``[W, control_dim]`` past controls ``u[i-W : i]``
  - ``state``        ``[state_dim]``      current state ``x[i]``
  - ``control``      ``[control_dim]``    current control ``u[i]``
  - ``next_state``   ``[state_dim]``      target ``x[i+1]``

  so a valid current index ``i`` runs over ``[W, T - 2]`` (needs ``W`` past
  steps and one future step). A :class:`~torch.utils.data.DataLoader` then adds
  the leading ``batch`` axis, giving ``[batch, W, dim]`` / ``[batch, dim]``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

ArrayLike = Union[Tensor, "Sequence[Sequence[float]]"]


def generate_synthetic_trajectories(
    A_m: ArrayLike,
    B_m: ArrayLike,
    dt: float,
    n_trajectories: int,
    n_steps: int,
    control_dim: int,
    *,
    seed: int | None = None,
    control_scale: float = 1.0,
    init_scale: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Roll out synthetic ``(state, control)`` trajectories (forward Euler).

    Reuses the SAME dynamics as the inline co-training plant
    (``training.cotrain._synthetic_plant_step``): each step advances
    ``x_dot = A_m x + B_m u`` by one Euler step ``x <- x + dt * x_dot``. The
    control sequence is drawn i.i.d. ``N(0, control_scale^2)`` per step and the
    initial state ``N(0, init_scale^2)``, all from a single seeded generator so
    runs are reproducible.

    Args:
        A_m: ``[state_dim, state_dim]`` state matrix (tensor / array-like).
        B_m: ``[state_dim, control_dim]`` input matrix (tensor / array-like).
        dt: Euler integration timestep (> 0).
        n_trajectories: number of independent rollouts.
        n_steps: number of timesteps per rollout (>= 1). Both the state and the
            control sequence have length ``n_steps``; state ``[:, t]`` is the
            state *before* applying control ``[:, t]``.
        control_dim: control dimension (must equal ``B_m.shape[1]``).
        seed: optional RNG seed for reproducibility.
        control_scale: std of the sampled control noise.
        init_scale: std of the sampled initial state.

    Returns:
        ``(states, controls)`` with shapes ``[n_trajectories, n_steps,
        state_dim]`` and ``[n_trajectories, n_steps, control_dim]`` (float32).
    """
    A = torch.as_tensor(A_m, dtype=torch.float32)
    B = torch.as_tensor(B_m, dtype=torch.float32)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A_m must be square [n, n]; got {tuple(A.shape)}")
    state_dim = A.shape[0]
    if B.ndim != 2 or B.shape[0] != state_dim:
        raise ValueError(
            f"B_m must be [state_dim, control_dim]=[{state_dim}, *]; got {tuple(B.shape)}"
        )
    if B.shape[1] != control_dim:
        raise ValueError(f"control_dim={control_dim} must equal B_m.shape[1]={B.shape[1]}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive; got {dt}")
    if n_trajectories < 1 or n_steps < 1:
        raise ValueError(
            f"n_trajectories and n_steps must be >= 1; got {n_trajectories}, {n_steps}"
        )

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    controls = (
        torch.randn(n_trajectories, n_steps, control_dim, generator=generator) * control_scale
    )
    states = torch.empty(n_trajectories, n_steps, state_dim, dtype=torch.float32)
    x = torch.randn(n_trajectories, state_dim, generator=generator) * init_scale
    for t in range(n_steps):
        states[:, t, :] = x
        u = controls[:, t, :]
        x_dot = torch.einsum("ij,bj->bi", A, x) + torch.einsum("ij,bj->bi", B, u)
        x = x + dt * x_dot
    return states, controls


def _coerce_trajectories(
    states: "ArrayLike | Sequence[ArrayLike]",
    controls: "ArrayLike | Sequence[ArrayLike]",
) -> Tuple[Tensor, Tensor]:
    """Coerce ``(states, controls)`` to batched ``[n_traj, T, dim]`` float32 tensors.

    Accepts either:

    * a single trajectory: ``states`` ``[T, state_dim]`` + ``controls``
      ``[T, control_dim]`` (promoted to ``n_traj == 1``), or
    * a batch: ``[n_traj, T, state_dim]`` / ``[n_traj, T, control_dim]``, or
    * a Python list/sequence of per-trajectory ``[T, dim]`` tensors/arrays of
      equal ``T`` (stacked into a batch).

    Raises ``ValueError`` / ``TypeError`` on shape or type mismatch.
    """

    def to_batched(obj: object, name: str) -> Tensor:
        # A list/tuple of per-trajectory 2D arrays -> stack to 3D.
        if isinstance(obj, (list, tuple)):
            mats = [torch.as_tensor(o, dtype=torch.float32) for o in obj]
            if not mats:
                raise ValueError(f"{name}: empty trajectory list")
            for m in mats:
                if m.ndim != 2:
                    raise ValueError(
                        f"{name}: each trajectory must be 2D [T, dim]; got {tuple(m.shape)}"
                    )
            return torch.stack(mats, dim=0)
        t = torch.as_tensor(obj, dtype=torch.float32)
        if t.ndim == 2:  # single [T, dim] -> [1, T, dim]
            return t.unsqueeze(0)
        if t.ndim == 3:
            return t
        raise ValueError(
            f"{name}: expected 2D [T, dim] or 3D [n_traj, T, dim]; got {tuple(t.shape)}"
        )

    s = to_batched(states, "states")
    c = to_batched(controls, "controls")
    if s.shape[0] != c.shape[0]:
        raise ValueError(f"states/controls trajectory count mismatch: {s.shape[0]} vs {c.shape[0]}")
    if s.shape[1] != c.shape[1]:
        raise ValueError(f"states/controls timestep length mismatch: {s.shape[1]} vs {c.shape[1]}")
    return s, c


class TrajectoryDataset(Dataset):
    """Windowed ``(state, control)`` trajectory dataset for the PITNN.

    Holds one or more trajectories and exposes the *windowed* samples that the
    :class:`~pits_mras.models.pitnn.PITNN` consumes: for each valid current
    index ``i`` in each trajectory it yields the ``memory_horizon``-step history
    window plus the current state/control and the next-state target. See the
    module docstring for the exact per-sample shapes.

    Args:
        states: trajectory states -- a single ``[T, state_dim]``, a batch
            ``[n_traj, T, state_dim]``, or a list of ``[T, state_dim]``.
        controls: matching controls (same outer/length structure as ``states``).
        memory_horizon: history-window length ``W`` (>= 1). Each trajectory must
            be long enough to yield at least one window (``T >= W + 2``).

    Raises:
        ValueError: on shape mismatch or trajectories too short to window.
    """

    def __init__(
        self,
        states: "ArrayLike | Sequence[ArrayLike]",
        controls: "ArrayLike | Sequence[ArrayLike]",
        *,
        memory_horizon: int,
    ) -> None:
        if memory_horizon < 1:
            raise ValueError(f"memory_horizon must be >= 1; got {memory_horizon}")
        s, c = _coerce_trajectories(states, controls)
        self.states: Tensor = s  # [n_traj, T, state_dim]
        self.controls: Tensor = c  # [n_traj, T, control_dim]
        self.memory_horizon = memory_horizon

        self.n_traj = s.shape[0]
        self.traj_len = s.shape[1]
        self.state_dim = s.shape[2]
        self.control_dim = c.shape[2]

        # Valid current indices per trajectory: i in [W, T-2] (need W past steps
        # and one future step i+1). n_windows = (T-1) - W.
        self._windows_per_traj = self.traj_len - 1 - memory_horizon
        if self._windows_per_traj < 1:
            raise ValueError(
                "trajectories too short to window: need traj_len >= memory_horizon + 2 "
                f"(traj_len={self.traj_len}, memory_horizon={memory_horizon})"
            )

        # Flat (traj, current-index) sample index map.
        self._index: List[Tuple[int, int]] = [
            (k, i) for k in range(self.n_traj) for i in range(memory_horizon, self.traj_len - 1)
        ]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        k, i = self._index[idx]
        W = self.memory_horizon
        return {
            "state_hist": self.states[k, i - W : i, :],  # [W, state_dim]
            "control_hist": self.controls[k, i - W : i, :],  # [W, control_dim]
            "state": self.states[k, i, :],  # [state_dim]
            "control": self.controls[k, i, :],  # [control_dim]
            "next_state": self.states[k, i + 1, :],  # [state_dim]
        }


def make_dataloader(
    dataset: TrajectoryDataset,
    batch_size: int,
    *,
    shuffle: bool = True,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Wrap a :class:`TrajectoryDataset` in a :class:`~torch.utils.data.DataLoader`.

    Uses the default collate, which stacks the per-sample dict fields into
    batched tensors: ``state_hist``/``control_hist`` become
    ``[batch, W, dim]`` and ``state``/``control``/``next_state`` become
    ``[batch, dim]`` -- exactly the leading-``batch`` layout the PITNN expects.

    Args:
        dataset: the trajectory dataset to iterate.
        batch_size: samples per batch (>= 1).
        shuffle: shuffle sample order each epoch (default ``True``).
        drop_last: drop a trailing partial batch (default ``False``).
        generator: optional RNG generator for reproducible shuffling.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator,
    )
