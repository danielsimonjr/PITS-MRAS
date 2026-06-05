r"""Adaptive / causal loss weighting utilities (ROADMAP #8).

Two cheap, opt-in techniques for balancing the multi-term PITNN objective. Both
operate on loss *values* (no extra backward passes), so they are inexpensive to
drop into the co-training loop.

ReLoBRaLo (Relative Loss Balancing with Random Lookback)
--------------------------------------------------------
Bischof & Kraus, "Multi-Objective Loss Balancing for Physics-Informed Deep
Learning" (arXiv:2110.09813). Given the per-term loss values ``L_i(t)`` at step
``t`` it produces balancing weights ``lambda_i`` that automatically up-weight
terms that are making *relatively* less progress than the others.

The exact rule implemented here (matching the paper, Eqs. 9-11):

    Let ``L_i(0)`` be the loss values at the FIRST call (saved once) and
    ``L_i(t-1)`` the values at the previous call. A Bernoulli(rho) draw ``b``
    per call selects the lookback reference

        t'  =  t-1   with probability rho      (recent step)
              =  0     with probability 1-rho    (initial step)

    The "balanced" weights are the temperature-scaled softmax of the
    relative-progress ratios, scaled by ``num_losses`` (m):

        lambda_i^bal(t, t') = m * softmax_i( L_i(t) / (temperature * L_i(t')) )

    These are EMA-combined with the running historical weights (alpha mixing,
    where the historical term itself is interpolated through the Bernoulli draw
    so a "recent" draw biases toward the previous balanced weights and an
    "initial" draw biases toward the all-ones baseline):

        rho_t = b                                    (Bernoulli outcome, 0 or 1)
        lambda_i(t) = alpha * (rho_t * lambda_i(t-1) + (1 - rho_t) * 1)
                      + (1 - alpha) * lambda_i^bal(t, t')

    On the FIRST call the initial losses are stored and all-ones weights are
    returned. Every returned weight vector sums to ``num_losses`` (the softmax
    sums to 1 and the all-ones/previous-weight terms each already sum to
    ``num_losses``), so the weighted objective keeps the same scale as the
    unweighted average ``mean_i L_i``.

Bernoulli reproducibility
-------------------------
The lookback draw uses ``torch.rand(generator=...)`` when a ``torch.Generator``
is supplied (so tests are bit-for-bit reproducible). When no generator is given
the choice is made by a deterministic internal call counter: a draw of
``rho`` is "recent" iff ``(call_index * something) ...`` — concretely we compare
a counter-derived uniform against ``rho`` so the sequence is fixed and
side-effect-free (no global RNG consumption, which would perturb the
characterization test if this were ever wired with the default branch).

causal_weights
--------------
Wang, Sankaran & Perdikaris, "Respecting Causality for Training Physics-Informed
Neural Networks" (arXiv:2203.07404). For per-timestep residual magnitudes the
weight of timestep ``i`` is suppressed by the *accumulated* residual of all
EARLIER timesteps:

    w_i = exp(-eps * sum_{k<i} residual_k)

so ``w_0 = 1`` and later weights shrink until the earlier residuals are small —
enforcing that a PINN learns the solution in temporal order. When ``all weights
-> 1`` the earlier residuals have vanished, which signals temporal convergence.
The weights are detached (a multiplier, not differentiated through).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

__all__ = ["ReLoBRaLo", "causal_weights"]

_EPS = 1e-12


class ReLoBRaLo:
    """Relative Loss Balancing with Random Lookback (arXiv:2110.09813).

    See the module docstring for the exact update rule. Cheap: consumes only the
    scalar loss values, no extra autograd.
    """

    def __init__(
        self,
        num_losses: int,
        alpha: float = 0.999,
        temperature: float = 1.0,
        rho: float = 0.9999,
    ) -> None:
        if num_losses < 1:
            raise ValueError(f"num_losses must be >= 1, got {num_losses}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not (0.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [0, 1], got {rho}")

        self.num_losses = num_losses
        self.alpha = alpha
        self.temperature = temperature
        self.rho = rho

        # Running state, lazily initialized on the first call.
        self._init_losses: Tensor | None = None  # L_i(0)
        self._prev_losses: Tensor | None = None  # L_i(t-1)
        self._hist_weights: Tensor | None = None  # lambda_i(t-1)
        self._call_count = 0

    def _to_tensor(self, losses: Sequence[float] | Tensor) -> Tensor:
        if isinstance(losses, Tensor):
            t = losses.detach().to(dtype=torch.float32).flatten()
        else:
            t = torch.tensor(list(losses), dtype=torch.float32)
        if t.numel() != self.num_losses:
            raise ValueError(f"expected {self.num_losses} loss values, got {t.numel()}")
        return t

    def _bernoulli(self, generator: torch.Generator | None) -> float:
        """Return the lookback Bernoulli outcome (1.0 = recent step, 0.0 = init).

        Reproducible: uses the supplied generator if any, else a deterministic
        counter-based pseudo-uniform that consumes NO global RNG state.
        """
        if generator is not None:
            u = float(torch.rand((), generator=generator).item())
        else:
            # Deterministic low-discrepancy-ish sequence from the call counter
            # (golden-ratio multiplier) — no global RNG side effects.
            u = (self._call_count * 0.6180339887498949) % 1.0
        return 1.0 if u < self.rho else 0.0

    def weights(
        self,
        losses: Sequence[float] | Tensor,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Return balancing weights (length ``num_losses``, summing to it)."""
        self._call_count += 1
        cur = self._to_tensor(losses)

        # First call: store the initial losses, return all-ones.
        if self._init_losses is None:
            self._init_losses = cur.clone()
            self._prev_losses = cur.clone()
            self._hist_weights = torch.ones(self.num_losses)
            return self._hist_weights.clone()

        assert self._init_losses is not None
        assert self._prev_losses is not None
        assert self._hist_weights is not None

        b = self._bernoulli(generator)
        # Lookback reference t': previous step (b=1) or initial step (b=0).
        ref = self._prev_losses if b == 1.0 else self._init_losses

        # Balanced weights: m * softmax_i(L_i(t) / (T * L_i(t'))).
        ratios = cur / (self.temperature * (ref + _EPS))
        lam_bal = self.num_losses * torch.softmax(ratios, dim=0)

        # Historical term interpolated through the same Bernoulli draw:
        # recent -> previous weights; initial -> all-ones baseline.
        hist_ref = self._hist_weights if b == 1.0 else torch.ones(self.num_losses)
        lam = self.alpha * hist_ref + (1.0 - self.alpha) * lam_bal

        # Renormalize defensively so the sum is EXACTLY num_losses (both the
        # convex parts already sum to num_losses, but float drift can creep in).
        lam = lam * (self.num_losses / (lam.sum() + _EPS))

        self._prev_losses = cur.clone()
        self._hist_weights = lam.clone()
        return lam


def causal_weights(residuals: Tensor, eps: float = 1.0) -> Tensor:
    r"""Causal training weights (arXiv:2203.07404).

    ``residuals``: per-timestep residual magnitudes, shape ``[T]`` or
    ``[batch, T]`` (time is the last axis, ordered). Returns

        w_i = exp(-eps * sum_{k<i} residual_k)

    (same shape as the input), so ``w_0 = 1`` and later weights shrink until the
    earlier residuals are small. ``all weights -> 1`` signals temporal
    convergence. The result is detached (a multiplier, not differentiated
    through).
    """
    r = residuals.detach()
    # Exclusive cumulative sum along the time axis (last): cumsum_{k<i}.
    inclusive = torch.cumsum(r, dim=-1)
    exclusive = inclusive - r  # shift so index 0 -> 0, index i -> sum_{k<i}
    return torch.exp(-eps * exclusive)
