r"""TD-MPC2 latent world model + sampling-based (MPPI) latent planner.

Owning phase: Phase 2 (Neural Network Models).

Implements the core of TD-MPC2 (Hansen et al. 2024, "TD-MPC2: Scalable, Robust
World Models for Continuous Control"): a learned latent world model and
sampling-based model-predictive control (MPPI / CEM) that plans in the latent
space. This is the learned-model-planning module the Blueprint lists as missing
(Connection 9); it is additive and self-contained.

* :class:`WorldModel` -- the TD-MPC2 components as MLP heads over a latent state:

  - ``encode(s) -> z``: observation/state encoder ``[B, state_dim] -> [B, d]``,
  - ``next(z, a) -> z'``: latent dynamics ``([B, d], [B, A]) -> [B, d]``,
  - ``reward(z, a) -> [B, 1]``: instantaneous reward predictor,
  - ``value(z) -> [B, 1]``: terminal state-value (the ``V`` head),
  - ``Q(z, a) -> [B, 1]``: optional latent action-value head.

  All inputs/outputs are ``[batch, dim]`` float tensors. The planner only needs
  ``next`` / ``reward`` / ``value``; ``Q`` supports the TD value update.

* :class:`MPPIPlanner` -- sampling-based MPC in latent space. It samples ``N``
  action SEQUENCES of length ``H`` from a per-step diagonal Gaussian, rolls each
  through the latent model, scores it by discounted predicted reward plus the
  discounted terminal ``value`` of the final latent, then iteratively re-fits the
  sampling distribution toward the score-weighted top-``k`` elite sequences (the
  MPPI exponential-weighting / CEM update). After ``iterations`` refinements it
  returns the planned FIRST action.

The latent planning loop is gradient-free (``torch.no_grad`` rollouts); the
world model is trained separately by :func:`pits_mras.training.tdmpc.tdmpc_update`.
"""

from __future__ import annotations

from typing import Optional, Protocol

import torch
import torch.nn as nn
from torch import Tensor


def _mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
    """Build a ReLU MLP ``in_dim -> hidden... -> out_dim`` (linear output head)."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class LatentModel(Protocol):
    """Structural interface the planner consumes (any object with these methods)."""

    def next(self, z: Tensor, a: Tensor) -> Tensor:  # noqa: D102
        ...

    def reward(self, z: Tensor, a: Tensor) -> Tensor:  # noqa: D102
        ...

    def value(self, z: Tensor) -> Tensor:  # noqa: D102
        ...


class WorldModel(nn.Module):
    r"""TD-MPC2 latent world model (encoder + dynamics + reward + value + Q).

    A shared latent dimension ``latent_dim`` ties the heads together. Each head is
    an independent ReLU MLP. Shapes (``B`` = batch):

    ===========  ==================  =================
    method       input               output
    ===========  ==================  =================
    ``encode``   ``[B, state_dim]``  ``[B, latent_dim]``
    ``next``     ``[B, d], [B, A]``  ``[B, latent_dim]``
    ``reward``   ``[B, d], [B, A]``  ``[B, 1]``
    ``value``    ``[B, latent_dim]`` ``[B, 1]``
    ``Q``        ``[B, d], [B, A]``  ``[B, 1]``
    ===========  ==================  =================

    (``d`` = ``latent_dim``, ``A`` = ``action_dim``.)

    Args:
        state_dim: dimension of the observation / state input.
        action_dim: dimension of the (continuous) action.
        latent_dim: dimension of the latent state ``z``.
        hidden: hidden-layer widths shared by every head (ReLU activations).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 32,
        hidden: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.encoder = _mlp(state_dim, latent_dim, hidden)
        self.dynamics = _mlp(latent_dim + action_dim, latent_dim, hidden)
        self.reward_head = _mlp(latent_dim + action_dim, 1, hidden)
        self.value_head = _mlp(latent_dim, 1, hidden)
        self.q_head = _mlp(latent_dim + action_dim, 1, hidden)

    def encode(self, s: Tensor) -> Tensor:
        r"""Encode a state ``[B, state_dim]`` into a latent ``z`` ``[B, latent_dim]``."""
        return self.encoder(s)

    def next(self, z: Tensor, a: Tensor) -> Tensor:
        r"""Predict the next latent ``z'`` ``[B, latent_dim]`` from ``(z, a)``."""
        return self.dynamics(torch.cat([z, a], dim=-1))

    def reward(self, z: Tensor, a: Tensor) -> Tensor:
        r"""Predict the instantaneous reward ``[B, 1]`` for latent-action ``(z, a)``."""
        return self.reward_head(torch.cat([z, a], dim=-1))

    def value(self, z: Tensor) -> Tensor:
        r"""Predict the terminal state-value ``V(z)`` ``[B, 1]``."""
        return self.value_head(z)

    def Q(self, z: Tensor, a: Tensor) -> Tensor:
        r"""Predict the latent action-value ``Q(z, a)`` ``[B, 1]``."""
        return self.q_head(torch.cat([z, a], dim=-1))

    def forward(self, s: Tensor, a: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        r"""Encode ``s`` then return ``(z_next, reward, value(z_next))`` for ``(s, a)``.

        Convenience differentiable path: ``z = encode(s)``, ``z' = next(z, a)``,
        and the predicted ``(z', reward(z, a), value(z'))``.
        """
        z = self.encode(s)
        z_next = self.next(z, a)
        return z_next, self.reward(z, a), self.value(z_next)


class MPPIPlanner:
    r"""Sampling-based latent MPC planner (MPPI / CEM with temperature).

    Given a latent model (a :class:`WorldModel` or any object exposing
    ``next`` / ``reward`` / ``value``) and an initial state, the planner:

    1. samples ``num_samples`` action sequences ``[N, H, action_dim]`` from a
       per-step diagonal Gaussian ``(mean[H, A], std[H, A])`` (clamped to the
       action bounds);
    2. rolls each sequence through the latent dynamics from the encoded initial
       state, accumulating discounted predicted reward
       ``sum_t gamma^t * reward(z_t, a_t)`` plus the discounted terminal value
       ``gamma^H * value(z_H)``;
    3. selects the top-``num_elites`` sequences by return and re-fits the Gaussian
       to the elites with MPPI exponential weighting
       ``w_i ∝ exp((R_i - R_max) / temperature)`` (the temperature -> 0 limit is
       hard CEM; larger temperature softens toward a uniform elite average);
    4. repeats for ``iterations`` refinements.

    Returns the planned FIRST action ``mean[0]`` shaped ``[1, action_dim]``.

    Args:
        action_dim: dimension of the continuous action.
        horizon: planning horizon ``H``.
        num_samples: number of action sequences sampled per iteration.
        iterations: number of distribution-refinement iterations.
        num_elites: number of top-return sequences kept per iteration.
        temperature: MPPI softmax temperature over elite returns (``> 0``).
        gamma: discount factor for the rollout return.
        action_low: lower action bound (scalar, applied per dim).
        action_high: upper action bound (scalar, applied per dim).
        init_std: initial per-step sampling standard deviation.
        min_std: floor on the refitted std for numerical stability / exploration.
    """

    def __init__(
        self,
        action_dim: int,
        *,
        horizon: int = 5,
        num_samples: int = 512,
        iterations: int = 6,
        num_elites: int = 64,
        temperature: float = 0.5,
        gamma: float = 0.99,
        action_low: float = -1.0,
        action_high: float = 1.0,
        init_std: float = 1.0,
        min_std: float = 1e-3,
    ) -> None:
        if num_elites > num_samples:
            raise ValueError("num_elites must not exceed num_samples")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.iterations = iterations
        self.num_elites = num_elites
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.init_std = float(init_std)
        self.min_std = float(min_std)

    @torch.no_grad()
    def plan(
        self,
        model: LatentModel,
        state: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        r"""Plan from a single initial ``state`` and return the first action.

        Args:
            model: latent model exposing ``next`` / ``reward`` / ``value``. If it
                also exposes ``encode`` the state is encoded to a latent first;
                otherwise ``state`` is treated as already-latent.
            state: initial state ``[1, state_dim]`` (or ``[1, latent_dim]`` if the
                model has no ``encode``).
            generator: optional :class:`torch.Generator` for reproducible sampling.

        Returns:
            The planned first action ``[1, action_dim]`` (clamped to bounds).
        """
        device = state.device
        dtype = state.dtype

        encode = getattr(model, "encode", None)
        z0 = encode(state) if callable(encode) else state  # [1, latent_dim]
        # Broadcast the single initial latent across all sampled sequences.
        z_init = z0.expand(self.num_samples, -1)

        mean = torch.zeros(self.horizon, self.action_dim, device=device, dtype=dtype)
        std = torch.full((self.horizon, self.action_dim), self.init_std, device=device, dtype=dtype)

        for _ in range(self.iterations):
            # Sample [N, H, A] action sequences from the per-step Gaussian.
            noise = torch.randn(
                self.num_samples,
                self.horizon,
                self.action_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            actions = actions.clamp(self.action_low, self.action_high)

            returns = self._rollout_returns(model, z_init, actions)  # [N]

            # Top-k elites by return.
            _, elite_idx = torch.topk(returns, self.num_elites)
            elite_returns = returns[elite_idx]  # [k]
            elite_actions = actions[elite_idx]  # [k, H, A]

            # MPPI exponential weighting (shift by max for numerical stability).
            weights = torch.softmax(
                (elite_returns - elite_returns.max()) / self.temperature, dim=0
            )  # [k]
            w = weights.view(self.num_elites, 1, 1)
            mean = (w * elite_actions).sum(dim=0)  # [H, A]
            var = (w * (elite_actions - mean.unsqueeze(0)) ** 2).sum(dim=0)
            std = var.sqrt().clamp_min(self.min_std)

        first_action = mean[0:1].clamp(self.action_low, self.action_high)
        return first_action  # [1, action_dim]

    def _rollout_returns(self, model: LatentModel, z_init: Tensor, actions: Tensor) -> Tensor:
        r"""Score ``[N, H, A]`` sequences: discounted reward + terminal value -> ``[N]``."""
        z = z_init
        total = torch.zeros(actions.shape[0], 1, device=actions.device, dtype=actions.dtype)
        discount = 1.0
        for t in range(self.horizon):
            a_t = actions[:, t, :]
            total = total + discount * model.reward(z, a_t)
            z = model.next(z, a_t)
            discount *= self.gamma
        total = total + discount * model.value(z)
        return total.squeeze(-1)


__all__ = ["WorldModel", "MPPIPlanner", "LatentModel"]
