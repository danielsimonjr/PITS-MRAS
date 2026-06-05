r"""Learned (neural) H-infinity adversary -- the disturbance policy (ROADMAP #1).

Owning phase: Phase 2 (Neural Network Models).

The analytic :class:`~pits_mras.models.critic.AdversaryHead` produces the
worst-case disturbance ``w* = gamma^-2 D^T P e`` BY CONSTRUCTION from the critic
gradient -- it is consistent with the critic but is not an independent learner.
The neural adversarial min-max loop (ROADMAP #1) replaces it with an INDEPENDENT
network :class:`NeuralAdversary` that is trained by gradient ASCENT on the game
value, so the protagonist (control) and critic must become robust to a learned
disturbance rather than an analytically-pinned one. At convergence the learned
policy should recover the analytic gain: ``w(e) -> L* e = gamma^-2 D^T P* e``.

This is an additive counterpart to ``AdversaryHead`` -- the analytic head is left
untouched and remains the oracle / warm-start reference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class NeuralAdversary(nn.Module):
    r"""Learned worst-case-disturbance policy ``w = pi_w(e)`` (an MLP).

    Counterpart to the analytic :class:`~pits_mras.models.critic.AdversaryHead`.
    Where the analytic head is the (scaled, ``D``-projected) critic gradient, this
    is an INDEPENDENT network trained by gradient ascent on the game value in the
    min-max loop. A plain Tanh-MLP maps the tracking error ``e`` to a disturbance
    ``w``; the output layer is linear (no squashing) so the loop can recover the
    linear analytic policy ``w*(e) = L* e`` exactly in the LTI regime.

    The output weights are initialized SMALL (near zero) so the adversary starts
    as a weak perturbation and grows during training -- this is a standard, gentle
    warm start that keeps the early closed loop stable while the critic settles.

    Args:
        state_dim: dimension of the tracking error ``e`` (the network input).
        dist_dim: dimension of the disturbance ``w`` (the network output).
        hidden: hidden-layer widths (Tanh activations between them).
    """

    def __init__(
        self,
        state_dim: int,
        dist_dim: int,
        hidden: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.dist_dim = dist_dim

        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        out = nn.Linear(in_dim, dist_dim)
        # Small init so the disturbance starts near zero (weak adversary).
        with torch.no_grad():
            out.weight.mul_(0.01)
            out.bias.zero_()
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, e: Tensor) -> Tensor:
        r"""Return the disturbance ``w`` of shape ``[batch, dist_dim]``.

        ``e`` has shape ``[batch, state_dim]``.
        """
        return self.net(e)
