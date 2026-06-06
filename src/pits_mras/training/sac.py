r"""Soft Actor-Critic update logic with automatic temperature (Connection 5).

Owning phase: Phase 5 (Training Pipelines).

Implements the SAC learner (Haarnoja et al. 2018, "Soft Actor-Critic" + the
automatic-temperature follow-up "Soft Actor-Critic Algorithms and
Applications"). The dedicated max-entropy-RL module the Blueprint lists as
missing (Connection 5); additive and self-contained — it consumes the actor /
critic networks from :mod:`pits_mras.models.sac` and owns the off-policy update.

:class:`SACTrainer` holds the squashed-Gaussian policy, the twin Q critic, its
target copy, three Adam optimizers, and a learnable log-temperature ``log_alpha``
tuned toward the target entropy ``-action_dim``. One :meth:`update` call performs
the three SAC losses on a transition batch and a soft (Polyak) target update.

Loss forms (entropy-regularized Bellman backup, clipped double-Q):

* **Critic target** ``y = r + gamma * (1 - done) * (min_target_Q(s', a') -
  alpha * log_prob(a' | s'))`` with ``a' ~ policy(s')``; critic loss
  ``= MSE(q1, y) + MSE(q2, y)`` (target detached).
* **Actor** ``L_pi = E[alpha * log_prob(a | s) - min_Q(s, a)]`` with
  ``a ~ policy(s)`` via the reparameterization trick.
* **Temperature** ``L_alpha = E[-log_alpha * (log_prob + target_entropy).detach()]``.
* **Targets** soft-updated ``theta_targ <- tau * theta + (1 - tau) * theta_targ``.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from pits_mras.models.sac import GaussianPolicy, TwinQCritic


class SACTrainer:
    r"""Soft Actor-Critic learner with automatic entropy temperature.

    Args:
        state_dim: state / observation dimension.
        action_dim: continuous action dimension.
        gamma: discount factor.
        tau: Polyak coefficient for the soft target update.
        lr: Adam learning rate shared by actor, critic, and temperature.
        target_entropy: entropy target for the temperature update; defaults to
            ``-action_dim`` (the SAC heuristic).
        hidden: hidden widths for both the policy and each Q-network.
        action_scale: env action range; forwarded to :class:`GaussianPolicy`.
        log_std_bounds: ``log_std`` clamp; forwarded to :class:`GaussianPolicy`.
        device: torch device (defaults to CPU).
        seed: optional RNG seed for reproducible init.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        target_entropy: Optional[float] = None,
        hidden: tuple[int, ...] = (256, 256),
        action_scale: float = 1.0,
        log_std_bounds: tuple[float, float] = (-20.0, 2.0),
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        self.device = device if device is not None else torch.device("cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = (
            float(target_entropy) if target_entropy is not None else -float(action_dim)
        )

        self.policy = GaussianPolicy(
            state_dim,
            action_dim,
            hidden=hidden,
            log_std_bounds=log_std_bounds,
            action_scale=action_scale,
        ).to(self.device)
        self.critic = TwinQCritic(state_dim, action_dim, hidden=hidden).to(self.device)
        self.critic_target = TwinQCritic(state_dim, action_dim, hidden=hidden).to(self.device)
        # Initialize target == online and freeze it from the optimizer.
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # Learnable log-temperature; alpha = exp(log_alpha) stays positive.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self) -> Tensor:
        """Current temperature ``alpha = exp(log_alpha)`` (positive scalar tensor)."""
        return self.log_alpha.exp()

    def _soft_update(self) -> None:
        """Polyak-average the target critic toward the online critic by ``tau``."""
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.mul_(1.0 - self.tau).add_(self.tau * p)

    def update(self, batch: Mapping[str, Tensor]) -> dict[str, float]:
        r"""Run one SAC update on a transition batch; return a metrics dict.

        Args:
            batch: mapping with keys ``state`` ``[B, state_dim]``, ``action``
                ``[B, action_dim]``, ``reward`` ``[B, 1]`` or ``[B]``,
                ``next_state`` ``[B, state_dim]``, ``done`` ``[B, 1]`` or ``[B]``.

        Returns:
            Dict of finite scalar losses ``critic_loss`` / ``actor_loss`` /
            ``alpha_loss``, the current ``alpha``, and the mean entropy proxy
            ``log_prob`` for monitoring.
        """
        state = batch["state"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device).reshape(-1, 1)
        next_state = batch["next_state"].to(self.device)
        done = batch["done"].to(self.device).reshape(-1, 1).to(state.dtype)

        # ── Critic update: entropy-regularized clipped double-Q target. ──
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state)
            target_q = self.critic_target.q_min(next_state, next_action)
            soft_target = target_q - self.alpha * next_log_prob
            y = reward + self.gamma * (1.0 - done) * soft_target

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── Actor update: reparameterized policy gradient. ──
        new_action, log_prob, _ = self.policy.sample(state)
        q_pi = self.critic.q_min(state, new_action)
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()
        self.policy_opt.zero_grad()
        actor_loss.backward()
        self.policy_opt.step()

        # ── Temperature update toward the target entropy. ──
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ── Soft target update. ──
        self._soft_update()

        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "alpha": float(self.alpha.detach()),
            "log_prob": float(log_prob.detach().mean()),
        }


__all__ = ["SACTrainer"]
