r"""Soft Actor-Critic networks: squashed Gaussian policy + twin Q critic.

Owning phase: Phase 2 (Neural Network Models).

Implements the actor / critic networks of Soft Actor-Critic (Haarnoja et al.
2018, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
with a Stochastic Actor", and the automatic-temperature follow-up "Soft
Actor-Critic Algorithms and Applications"). This is the dedicated
max-entropy-RL module the Blueprint lists as missing (Connection 5); it is a
self-contained, additive counterpart to the existing analytic/IRL machinery.

* :class:`GaussianPolicy` -- a tanh-squashed diagonal-Gaussian stochastic actor.
  The reparameterization trick (``mean + std * eps``) keeps the sample
  differentiable w.r.t. the policy parameters, and the change-of-variables
  correction ``log_prob -= sum(log(1 - tanh(u)^2 + eps))`` accounts for the tanh
  squashing so the returned ``log_prob`` is the exact log-density of the bounded
  action. ``action_scale`` maps the squashed ``[-1, 1]`` action to the env range.
* :class:`TwinQCritic` -- two independent Q-networks (clipped double-Q) with a
  ``q_min`` helper, the standard SAC critic that combats value over-estimation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# Numerical floor inside log(1 - tanh(u)^2 + eps) for the squash correction.
_LOG_PROB_EPS = 1e-6


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


class GaussianPolicy(nn.Module):
    r"""Tanh-squashed diagonal-Gaussian stochastic actor (SAC).

    A shared MLP trunk maps the state to a per-dimension ``mean`` and
    ``log_std``; :meth:`sample` draws a reparameterized action ``a = scale *
    tanh(mean + std * eps)`` and returns its exact log-density under the squashed
    distribution. ``log_std`` is clamped to ``log_std_bounds`` for stability.

    Args:
        state_dim: dimension of the state / observation input.
        action_dim: dimension of the (continuous) action.
        hidden: hidden-layer widths of the trunk (ReLU activations).
        log_std_bounds: ``(min, max)`` clamp on the predicted ``log_std``.
        action_scale: scalar mapping the squashed ``[-1, 1]`` action to the env
            range ``[-action_scale, action_scale]``.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: tuple[int, ...] = (256, 256),
        log_std_bounds: tuple[float, float] = (-20.0, 2.0),
        action_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min, self.log_std_max = log_std_bounds
        self.action_scale = float(action_scale)
        # Trunk emits [mean | log_std] -> 2 * action_dim outputs.
        self.net = _mlp(state_dim, 2 * action_dim, hidden)

    def _mean_log_std(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(mean, log_std)`` each ``[batch, action_dim]`` (log_std clamped)."""
        out = self.net(state)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        r"""Reparameterized sample with tanh squashing.

        Returns ``(action, log_prob, mean_action)``:

        * ``action`` -- ``[batch, action_dim]`` in ``[-scale, scale]``,
          differentiable w.r.t. the policy parameters (reparam trick).
        * ``log_prob`` -- ``[batch, 1]`` exact log-density of ``action`` under the
          squashed Gaussian, with the tanh change-of-variables correction
          ``-sum(log(1 - tanh(u)^2 + eps))`` and the ``log(scale)`` Jacobian term.
        * ``mean_action`` -- ``[batch, action_dim]`` deterministic ``tanh(mean) *
          scale`` action (greedy / evaluation path).
        """
        mean, log_std = self._mean_log_std(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()  # reparameterized pre-squash sample
        tanh_u = torch.tanh(u)
        action = tanh_u * self.action_scale

        # log N(u) summed over action dims.
        log_prob = normal.log_prob(u).sum(dim=-1, keepdim=True)
        # Tanh squash correction + constant log(scale) Jacobian per dim.
        correction = torch.log(1.0 - tanh_u.pow(2) + _LOG_PROB_EPS).sum(dim=-1, keepdim=True)
        log_prob = log_prob - correction
        log_prob = log_prob - self.action_dim * torch.log(
            torch.tensor(self.action_scale, dtype=action.dtype, device=action.device)
        )

        mean_action = torch.tanh(mean) * self.action_scale
        return action, log_prob, mean_action

    def mean(self, state: Tensor) -> Tensor:
        r"""Deterministic greedy action ``tanh(mean) * scale`` ``[batch, action_dim]``."""
        mean, _ = self._mean_log_std(state)
        return torch.tanh(mean) * self.action_scale


class TwinQCritic(nn.Module):
    r"""Twin (clipped double-Q) state-action critic for SAC.

    Two INDEPENDENT Q-networks ``Q(s, a) -> [batch, 1]``; :meth:`forward` returns
    both ``(q1, q2)`` and :meth:`q_min` the elementwise minimum, used in the SAC
    target to mitigate positive value over-estimation bias.

    Args:
        state_dim: dimension of the state input.
        action_dim: dimension of the action input.
        hidden: hidden-layer widths of each Q-network (ReLU activations).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q1 = _mlp(state_dim + action_dim, 1, hidden)
        self.q2 = _mlp(state_dim + action_dim, 1, hidden)

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        r"""Return ``(q1, q2)`` each ``[batch, 1]`` for state-action ``(s, a)``."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q_min(self, state: Tensor, action: Tensor) -> Tensor:
        r"""Return ``min(q1, q2)`` elementwise, shape ``[batch, 1]``."""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


__all__ = ["GaussianPolicy", "TwinQCritic"]
