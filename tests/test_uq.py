"""Unit tests for ``pits_mras.utils.uq`` (uncertainty quantification).

Covers deep-ensemble epistemic uncertainty, split-conformal prediction
quantiles/intervals, and the online Adaptive Conformal Inference (ACI) updater.
"""

import math

import torch

from pits_mras.utils.uq import (
    AdaptiveConformalInference,
    DeepEnsemble,
    conformal_interval,
    split_conformal_quantile,
)


# ---------------------------------------------------------------------------
# DeepEnsemble
# ---------------------------------------------------------------------------
def test_deep_ensemble_empty_members_raises() -> None:
    """Constructing an ensemble with no members is an error."""
    try:
        DeepEnsemble([])
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty members")


def test_deep_ensemble_predict_all_shape() -> None:
    """``predict_all`` stacks K member outputs into ``[K, batch, d]``."""
    k, batch, d = 3, 5, 2

    def make_member(bias: float):
        return lambda x: x[:, :d] + bias

    members = [make_member(float(i)) for i in range(k)]
    ens = DeepEnsemble(members)
    x = torch.randn(batch, d)
    out = ens.predict_all(x)
    assert out.shape == (k, batch, d)


def test_deep_ensemble_mean_zero_std_for_identical_members() -> None:
    """Identical constant members give exact mean and zero std."""
    batch, d = 4, 3
    const = torch.full((batch, d), 7.0)
    members = [lambda x, c=const: c for _ in range(5)]
    ens = DeepEnsemble(members)
    x = torch.randn(batch, d)
    mean, std = ens.mean_and_std(x)
    torch.testing.assert_close(mean, const)
    torch.testing.assert_close(std, torch.zeros(batch, d))


def test_deep_ensemble_mean_and_std_hand_computed() -> None:
    """Mean/std match hand-computed population statistics for 3 members."""
    batch, d = 1, 1
    vals = [1.0, 2.0, 6.0]  # mean 3.0; population var = ((1-3)^2+(2-3)^2+(6-3)^2)/3
    members = [lambda x, v=v: torch.full((batch, d), v) for v in vals]
    ens = DeepEnsemble(members)
    x = torch.zeros(batch, d)
    mean, std = ens.mean_and_std(x)
    expected_mean = torch.full((batch, d), 3.0)
    pop_var = ((1 - 3) ** 2 + (2 - 3) ** 2 + (6 - 3) ** 2) / 3.0
    expected_std = torch.full((batch, d), math.sqrt(pop_var))
    torch.testing.assert_close(mean, expected_mean)
    torch.testing.assert_close(std, expected_std)


def test_deep_ensemble_mean_and_std_shapes() -> None:
    """Mean and std are each ``[batch, d]``."""
    batch, d = 6, 4
    members = [lambda x: x for _ in range(3)]
    ens = DeepEnsemble(members)
    x = torch.randn(batch, d)
    mean, std = ens.mean_and_std(x)
    assert mean.shape == (batch, d)
    assert std.shape == (batch, d)


# ---------------------------------------------------------------------------
# split_conformal_quantile
# ---------------------------------------------------------------------------
def test_split_conformal_quantile_known_order_statistic() -> None:
    """Returns the ceil((n+1)(1-alpha))/n empirical quantile (order statistic)."""
    # n=9, alpha=0.2 -> ceil(10*0.8)=8 -> the 8th smallest score.
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    q = split_conformal_quantile(scores, alpha=0.2)
    torch.testing.assert_close(q, torch.tensor(8.0))


def test_split_conformal_quantile_unsorted_input() -> None:
    """Works regardless of input ordering."""
    scores = torch.tensor([9.0, 1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0])
    q = split_conformal_quantile(scores, alpha=0.2)
    torch.testing.assert_close(q, torch.tensor(8.0))


def test_split_conformal_quantile_alpha_too_small_returns_inf() -> None:
    """If ceil((n+1)(1-alpha)) > n the interval is unbounded (+inf)."""
    scores = torch.arange(1.0, 11.0)  # n=10
    # alpha=0.01 -> ceil(11*0.99)=ceil(10.89)=11 > 10 -> +inf
    q = split_conformal_quantile(scores, alpha=0.01)
    assert math.isinf(q.item()) and q.item() > 0


def test_split_conformal_coverage_sanity() -> None:
    """Empirical coverage on a held-out i.i.d. set is approximately >= 1-alpha."""
    torch.manual_seed(0)
    alpha = 0.1
    n_cal = 2000
    cal_resid = torch.randn(n_cal).abs()
    q = split_conformal_quantile(cal_resid, alpha=alpha)
    test_resid = torch.randn(5000).abs()
    coverage = (test_resid <= q).float().mean().item()
    # Allow a little slack below the nominal level for finite-sample noise.
    assert coverage >= (1 - alpha) - 0.02


# ---------------------------------------------------------------------------
# conformal_interval
# ---------------------------------------------------------------------------
def test_conformal_interval_symmetric() -> None:
    """Returns (pred - q, pred + q)."""
    pred = torch.tensor([0.0, 1.0, -2.0])
    q = torch.tensor(0.5)
    lo, hi = conformal_interval(pred, q)
    torch.testing.assert_close(lo, pred - 0.5)
    torch.testing.assert_close(hi, pred + 0.5)


def test_conformal_interval_broadcasting() -> None:
    """Scalar half-width broadcasts against a batched prediction tensor."""
    pred = torch.randn(4, 2)
    q = torch.tensor(1.5)
    lo, hi = conformal_interval(pred, q)
    assert lo.shape == pred.shape
    assert hi.shape == pred.shape
    torch.testing.assert_close(hi - lo, torch.full_like(pred, 3.0))


# ---------------------------------------------------------------------------
# AdaptiveConformalInference
# ---------------------------------------------------------------------------
def test_aci_invalid_target_alpha_raises() -> None:
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            AdaptiveConformalInference(target_alpha=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for target_alpha={bad}")


def test_aci_invalid_gamma_raises() -> None:
    for bad in (0.0, -0.01):
        try:
            AdaptiveConformalInference(target_alpha=0.1, gamma=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for gamma={bad}")


def test_aci_initial_alpha_is_target() -> None:
    aci = AdaptiveConformalInference(target_alpha=0.1)
    assert aci.current_alpha == 0.1


def test_aci_all_covered_drives_alpha_up() -> None:
    """A run of covered observations raises alpha_t (looser -> tighter intervals)."""
    aci = AdaptiveConformalInference(target_alpha=0.1, gamma=0.01)
    start = aci.current_alpha
    for _ in range(20):
        aci.update(covered=True)
    assert aci.current_alpha > start


def test_aci_all_missed_drives_alpha_down() -> None:
    """A run of misses lowers alpha_t (smaller -> wider intervals)."""
    aci = AdaptiveConformalInference(target_alpha=0.1, gamma=0.01)
    start = aci.current_alpha
    for _ in range(20):
        aci.update(covered=False)
    assert aci.current_alpha < start


def test_aci_alpha_stays_clamped() -> None:
    """alpha_t never leaves [0, 1] under sustained one-sided feedback."""
    aci = AdaptiveConformalInference(target_alpha=0.5, gamma=0.5)
    for _ in range(100):
        aci.update(covered=True)
    assert 0.0 <= aci.current_alpha <= 1.0
    aci2 = AdaptiveConformalInference(target_alpha=0.5, gamma=0.5)
    for _ in range(100):
        aci2.update(covered=False)
    assert 0.0 <= aci2.current_alpha <= 1.0


def test_aci_update_returns_new_alpha() -> None:
    aci = AdaptiveConformalInference(target_alpha=0.1, gamma=0.01)
    returned = aci.update(covered=True)
    assert returned == aci.current_alpha


def test_aci_average_error_tracks_target() -> None:
    """Over a long stationary sequence the realized miscoverage tracks target.

    We simulate a Bernoulli miscoverage process whose probability responds to
    alpha_t: a larger alpha_t (tighter nominal interval) yields more misses.
    The ACI feedback should pull the running miss-rate toward target_alpha.
    """
    torch.manual_seed(0)
    target = 0.1
    aci = AdaptiveConformalInference(target_alpha=target, gamma=0.02)
    misses = 0
    steps = 5000
    for _ in range(steps):
        # Miss probability increases with alpha_t (proxy for tighter intervals).
        p_miss = min(max(aci.current_alpha, 0.0), 1.0)
        miss = torch.rand(1).item() < p_miss
        misses += int(miss)
        aci.update(covered=not miss)
    avg_err = misses / steps
    assert abs(avg_err - target) < 0.03
