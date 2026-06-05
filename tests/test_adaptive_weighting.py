"""Tests for adaptive / causal loss weighting utilities (ROADMAP #8).

Covers:
  * ReLoBRaLo — Relative Loss Balancing with Random Lookback (arXiv:2110.09813):
    argument validation, all-ones first call, sum-to-num_losses invariant,
    high-loss-term weight escalation, reproducibility under a fixed generator.
  * causal_weights — causal training weights (arXiv:2203.07404): w_0 == 1,
    monotone non-increasing in time, all-zero -> all-ones, early-residual
    suppression, batch-shape preservation, hand-computed small example.
"""

from __future__ import annotations

import math

import pytest
import torch

from pits_mras.losses.adaptive_weighting import ReLoBRaLo, causal_weights


# --------------------------------------------------------------------------- #
# ReLoBRaLo — argument validation.
# --------------------------------------------------------------------------- #
def test_relobralo_rejects_zero_losses() -> None:
    with pytest.raises(ValueError):
        ReLoBRaLo(num_losses=0)


def test_relobralo_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        ReLoBRaLo(num_losses=3, alpha=1.5)
    with pytest.raises(ValueError):
        ReLoBRaLo(num_losses=3, alpha=-0.1)


def test_relobralo_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError):
        ReLoBRaLo(num_losses=3, temperature=0.0)
    with pytest.raises(ValueError):
        ReLoBRaLo(num_losses=3, temperature=-1.0)


def test_relobralo_rejects_bad_rho() -> None:
    with pytest.raises(ValueError):
        ReLoBRaLo(num_losses=3, rho=1.5)
    with pytest.raises(ValueError):
        ReLoBRaLo(num_losses=3, rho=-0.1)


# --------------------------------------------------------------------------- #
# ReLoBRaLo — first call and sum invariant.
# --------------------------------------------------------------------------- #
def test_relobralo_first_call_all_ones() -> None:
    balancer = ReLoBRaLo(num_losses=3)
    w = balancer.weights([1.0, 2.0, 3.0])
    assert torch.allclose(w, torch.ones(3))


def test_relobralo_first_call_sums_to_num_losses() -> None:
    balancer = ReLoBRaLo(num_losses=4)
    w = balancer.weights([0.5, 0.5, 0.5, 0.5])
    assert math.isclose(float(w.sum()), 4.0, abs_tol=1e-5)


def test_relobralo_weights_always_sum_to_num_losses() -> None:
    gen = torch.Generator().manual_seed(123)
    balancer = ReLoBRaLo(num_losses=3)
    losses = [
        [1.0, 1.0, 1.0],
        [0.9, 0.5, 1.0],
        [0.8, 0.25, 1.0],
        [0.7, 0.1, 1.0],
        [0.6, 0.05, 1.0],
    ]
    for step in losses:
        w = balancer.weights(step, generator=gen)
        assert math.isclose(float(w.sum()), 3.0, abs_tol=1e-4)
        assert torch.all(w >= 0.0)


def test_relobralo_accepts_tensor_input() -> None:
    balancer = ReLoBRaLo(num_losses=2)
    w = balancer.weights(torch.tensor([1.0, 2.0]))
    assert w.shape == (2,)
    assert torch.allclose(w, torch.ones(2))


def test_relobralo_length_mismatch_raises() -> None:
    balancer = ReLoBRaLo(num_losses=3)
    with pytest.raises(ValueError):
        balancer.weights([1.0, 2.0])


# --------------------------------------------------------------------------- #
# ReLoBRaLo — stuck-high term gets up-weighted.
# --------------------------------------------------------------------------- #
def test_relobralo_stuck_term_increases_relative_weight() -> None:
    """When term 0 stays high while terms 1 and 2 drop, term 0's weight rises."""
    gen = torch.Generator().manual_seed(0)
    balancer = ReLoBRaLo(num_losses=3, alpha=0.5, rho=1.0, temperature=1.0)
    balancer.weights([1.0, 1.0, 1.0], generator=gen)  # init
    w_last = torch.ones(3)
    for _ in range(20):
        # term 0 stays high, terms 1 & 2 decay
        w_last = balancer.weights([1.0, 0.2, 0.2], generator=gen)
    # The stuck-high term should carry the largest weight.
    assert float(w_last[0]) > float(w_last[1])
    assert float(w_last[0]) > float(w_last[2])
    assert float(w_last[0]) > 1.0  # above the unweighted-average baseline


def test_relobralo_reproducible_with_fixed_generator() -> None:
    def run() -> torch.Tensor:
        gen = torch.Generator().manual_seed(42)
        b = ReLoBRaLo(num_losses=3, rho=0.5)
        b.weights([1.0, 1.0, 1.0], generator=gen)
        out = torch.zeros(3)
        for step in [[0.9, 1.0, 0.8], [0.7, 1.0, 0.6], [0.5, 1.0, 0.4]]:
            out = b.weights(step, generator=gen)
        return out

    assert torch.allclose(run(), run())


# --------------------------------------------------------------------------- #
# causal_weights.
# --------------------------------------------------------------------------- #
def test_causal_weights_first_is_one() -> None:
    res = torch.tensor([0.5, 0.3, 0.2, 0.1])
    w = causal_weights(res)
    assert math.isclose(float(w[0]), 1.0, abs_tol=1e-6)


def test_causal_weights_monotone_nonincreasing() -> None:
    res = torch.tensor([0.5, 0.3, 0.2, 0.1, 0.4])
    w = causal_weights(res)
    assert torch.all(w[1:] <= w[:-1] + 1e-6)


def test_causal_weights_all_zero_residual_all_ones() -> None:
    res = torch.zeros(6)
    w = causal_weights(res)
    assert torch.allclose(w, torch.ones(6))


def test_causal_weights_large_early_residual_suppresses_later() -> None:
    res = torch.tensor([10.0, 0.1, 0.1, 0.1])
    w = causal_weights(res, eps=1.0)
    assert math.isclose(float(w[0]), 1.0, abs_tol=1e-6)
    assert float(w[1]) < 1e-3  # suppressed by the large early residual


def test_causal_weights_batched_shape_preserved() -> None:
    res = torch.rand(5, 7).abs()
    w = causal_weights(res)
    assert w.shape == (5, 7)
    assert torch.allclose(w[:, 0], torch.ones(5))
    assert torch.all(w[:, 1:] <= w[:, :-1] + 1e-6)


def test_causal_weights_hand_computed_example() -> None:
    res = torch.tensor([1.0, 2.0, 3.0])
    eps = 0.5
    w = causal_weights(res, eps=eps)
    # w_i = exp(-eps * cumsum_{k<i} res_k): cumsum-exclusive = [0, 1, 3]
    expected = torch.exp(-eps * torch.tensor([0.0, 1.0, 3.0]))
    assert torch.allclose(w, expected, atol=1e-6)


def test_causal_weights_detached() -> None:
    res = torch.tensor([0.5, 0.3, 0.2], requires_grad=True)
    w = causal_weights(res)
    assert not w.requires_grad
