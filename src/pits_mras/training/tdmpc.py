r"""TD-MPC2 world-model update (Connection 9 — learned-model planning).

Owning phase: Phase 5 (Training Pipelines).

Implements the (bounded) TD-MPC2 world-model objective (Hansen et al. 2024,
"TD-MPC2: Scalable, Robust World Models for Continuous Control"). Additive and
self-contained: it consumes a :class:`pits_mras.models.tdmpc.WorldModel` and runs
one joint gradient step on the three core latent-model losses over a transition
batch.

:func:`tdmpc_update` computes, on a batch of ``(s, a, r, s', done)``:

* **Latent consistency** ``||next(encode(s), a) - sg[encode(s')]||^2`` -- the
  predicted next latent must match the (stop-gradient) encoding of the true next
  state, the self-supervised signal that grounds the latent dynamics.
* **Reward prediction** ``MSE(reward(z, a), r)``.
* **TD value** ``MSE(Q(z, a), y)`` with the one-step bootstrap target
  ``y = r + gamma * (1 - done) * value(sg[encode(s')])`` (target detached).

The three terms are summed (with weights) into one scalar and stepped through the
supplied optimizer. Returns a dict of finite scalar losses for monitoring.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from pits_mras.models.tdmpc import WorldModel


def tdmpc_update(
    model: WorldModel,
    batch: Mapping[str, Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    gamma: float = 0.99,
    consistency_coef: float = 1.0,
    reward_coef: float = 1.0,
    value_coef: float = 1.0,
) -> dict[str, float]:
    r"""Run one TD-MPC2 world-model update; return a metrics dict.

    Args:
        model: the :class:`WorldModel` to update.
        batch: mapping with keys ``state`` ``[B, state_dim]``, ``action``
            ``[B, action_dim]``, ``reward`` ``[B, 1]`` or ``[B]``, ``next_state``
            ``[B, state_dim]``, ``done`` ``[B, 1]`` or ``[B]``.
        optimizer: optimizer over ``model.parameters()``.
        gamma: discount factor in the TD value target.
        consistency_coef: weight on the latent-consistency loss.
        reward_coef: weight on the reward-prediction loss.
        value_coef: weight on the TD value loss.

    Returns:
        Dict of finite scalar losses ``consistency_loss`` / ``reward_loss`` /
        ``value_loss`` / ``total_loss``.
    """
    state = batch["state"]
    action = batch["action"]
    reward = batch["reward"].reshape(-1, 1).to(state.dtype)
    next_state = batch["next_state"]
    done = batch["done"].reshape(-1, 1).to(state.dtype)

    z = model.encode(state)
    z_next_pred = model.next(z, action)

    # Stop-gradient target encoding of the true next state.
    with torch.no_grad():
        z_next_target = model.encode(next_state)
        td_target = reward + gamma * (1.0 - done) * model.value(z_next_target)

    consistency_loss = F.mse_loss(z_next_pred, z_next_target)
    reward_loss = F.mse_loss(model.reward(z, action), reward)
    value_loss = F.mse_loss(model.Q(z, action), td_target)

    total_loss = (
        consistency_coef * consistency_loss + reward_coef * reward_loss + value_coef * value_loss
    )

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "consistency_loss": float(consistency_loss.detach()),
        "reward_loss": float(reward_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "total_loss": float(total_loss.detach()),
    }


__all__ = ["tdmpc_update"]
