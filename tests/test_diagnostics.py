"""Unit tests for ``pits_mras.utils.diagnostics`` (ROADMAP #4).

Long-horizon rollout-stability + conservation-drift diagnostics. Covers
``energy_drift`` / ``max_energy_drift`` (conservation-drift series),
``valid_prediction_time`` (VPT horizon), and
``rollout_jacobian_spectral_radius`` (one-step error-amplification factor).
"""

import torch

from pits_mras.utils.diagnostics import (
    energy_drift,
    max_energy_drift,
    rollout_jacobian_spectral_radius,
    valid_prediction_time,
)


# --------------------------------------------------------------------------- #
# energy_drift                                                                 #
# --------------------------------------------------------------------------- #
def test_energy_drift_constant_series_is_zero() -> None:
    """A perfectly conserved quantity has zero drift everywhere."""
    q = torch.full((6,), 3.0)
    drift = energy_drift(q, relative=False)
    assert drift.shape == (6,)
    torch.testing.assert_close(drift, torch.zeros(6))


def test_energy_drift_first_entry_is_zero() -> None:
    """drift[..., 0] is always exactly zero (q_0 - q_0)."""
    q = torch.tensor([2.0, 5.0, 9.0, -1.0])
    drift_abs = energy_drift(q, relative=False)
    drift_rel = energy_drift(q, relative=True)
    torch.testing.assert_close(drift_abs[0], torch.tensor(0.0))
    torch.testing.assert_close(drift_rel[0], torch.tensor(0.0))


def test_energy_drift_absolute_known_ramp() -> None:
    """Absolute drift of a linear ramp equals q_t - q_0."""
    q = torch.tensor([10.0, 11.0, 13.0, 16.0])
    drift = energy_drift(q, relative=False)
    expected = torch.tensor([0.0, 1.0, 3.0, 6.0])
    torch.testing.assert_close(drift, expected)


def test_energy_drift_relative_known_ramp() -> None:
    """Relative drift divides absolute drift by |q_0| + eps."""
    q = torch.tensor([10.0, 11.0, 13.0, 16.0])
    eps = 1e-8
    drift = energy_drift(q, relative=True, eps=eps)
    expected = torch.tensor([0.0, 1.0, 3.0, 6.0]) / (10.0 + eps)
    torch.testing.assert_close(drift, expected)


def test_energy_drift_batched_shape_preserved() -> None:
    """Batched [batch, T] input keeps its leading shape."""
    q = torch.tensor([[1.0, 2.0, 4.0], [5.0, 5.0, 5.0]])
    drift = energy_drift(q, relative=False)
    assert drift.shape == (2, 3)
    expected = torch.tensor([[0.0, 1.0, 3.0], [0.0, 0.0, 0.0]])
    torch.testing.assert_close(drift, expected)
    # First column zero for every batch element.
    torch.testing.assert_close(drift[:, 0], torch.zeros(2))


def test_energy_drift_relative_batched() -> None:
    """Relative drift normalizes per-batch by that row's q_0."""
    q = torch.tensor([[2.0, 4.0], [10.0, 5.0]])
    eps = 1e-8
    drift = energy_drift(q, relative=True, eps=eps)
    expected = torch.tensor([[0.0, 2.0 / (2.0 + eps)], [0.0, -5.0 / (10.0 + eps)]])
    torch.testing.assert_close(drift, expected)


# --------------------------------------------------------------------------- #
# max_energy_drift                                                             #
# --------------------------------------------------------------------------- #
def test_max_energy_drift_known_series() -> None:
    """Max absolute drift picks the largest |q_t - q_0|."""
    q = torch.tensor([10.0, 11.0, 8.0, 16.0])  # drifts: 0, 1, -2, 6
    m = max_energy_drift(q, relative=False)
    torch.testing.assert_close(m, torch.tensor(6.0))


def test_max_energy_drift_picks_negative_extreme() -> None:
    """The max is over absolute drift, so a big negative excursion wins."""
    q = torch.tensor([10.0, 9.0, 1.0, 12.0])  # drifts: 0, -1, -9, 2
    m = max_energy_drift(q, relative=False)
    torch.testing.assert_close(m, torch.tensor(9.0))


def test_max_energy_drift_relative() -> None:
    """Relative max drift divides by |q_0| + eps."""
    q = torch.tensor([10.0, 11.0, 8.0, 16.0])
    eps = 1e-8
    m = max_energy_drift(q, relative=True, eps=eps)
    torch.testing.assert_close(m, torch.tensor(6.0 / (10.0 + eps)))


def test_max_energy_drift_batched() -> None:
    """Batched input returns one max drift per batch element."""
    q = torch.tensor([[10.0, 11.0, 8.0, 16.0], [5.0, 5.0, 5.0, 5.0]])
    m = max_energy_drift(q, relative=False)
    assert m.shape == (2,)
    torch.testing.assert_close(m, torch.tensor([6.0, 0.0]))


# --------------------------------------------------------------------------- #
# valid_prediction_time                                                        #
# --------------------------------------------------------------------------- #
def test_vpt_perfect_prediction_full_horizon() -> None:
    """pred == truth never exceeds threshold -> VPT == (T-1)*dt."""
    truth = torch.randn(8, 3)
    pred = truth.clone()
    vpt = valid_prediction_time(pred, truth, threshold=0.1, dt=1.0)
    torch.testing.assert_close(vpt, torch.tensor(7.0))


def test_vpt_crosses_at_known_step() -> None:
    """Normalized error first exceeds threshold at a known t -> t*dt."""
    # truth has unit norm rows; pred matches except at t=3 where error blows up.
    truth = torch.zeros(5, 2)
    truth[:, 0] = 1.0  # ||truth_t|| = 1 everywhere
    pred = truth.clone()
    pred[3, 0] = 1.0 + 0.5  # error norm 0.5 at t=3 -> exceeds threshold 0.2
    vpt = valid_prediction_time(pred, truth, threshold=0.2, dt=1.0)
    torch.testing.assert_close(vpt, torch.tensor(3.0))


def test_vpt_dt_scaling() -> None:
    """VPT scales linearly with dt."""
    truth = torch.zeros(5, 2)
    truth[:, 0] = 1.0
    pred = truth.clone()
    pred[3, 0] = 1.0 + 0.5
    vpt = valid_prediction_time(pred, truth, threshold=0.2, dt=0.25)
    torch.testing.assert_close(vpt, torch.tensor(0.75))


def test_vpt_perfect_dt_scaling() -> None:
    """Full-horizon VPT also scales with dt: (T-1)*dt."""
    truth = torch.randn(6, 4)
    pred = truth.clone()
    vpt = valid_prediction_time(pred, truth, threshold=0.1, dt=0.5)
    torch.testing.assert_close(vpt, torch.tensor(2.5))


def test_vpt_batched_returns_per_batch() -> None:
    """Batched [batch, T, dim] input returns a [batch] tensor of VPTs."""
    truth = torch.zeros(2, 5, 2)
    truth[..., 0] = 1.0
    pred = truth.clone()
    # batch 0 crosses at t=2; batch 1 never crosses (full horizon).
    pred[0, 2, 0] = 1.0 + 0.5
    vpt = valid_prediction_time(pred, truth, threshold=0.2, dt=1.0)
    assert vpt.shape == (2,)
    torch.testing.assert_close(vpt, torch.tensor([2.0, 4.0]))


# --------------------------------------------------------------------------- #
# rollout_jacobian_spectral_radius                                            #
# --------------------------------------------------------------------------- #
def test_spectral_radius_diagonal_linear_map() -> None:
    """For step_fn(x) = A x with A = diag([0.5, 2.0]), radius = max|eig| = 2."""
    A = torch.diag(torch.tensor([0.5, 2.0]))

    def step_fn(x: torch.Tensor) -> torch.Tensor:
        return A @ x

    x = torch.tensor([1.0, 1.0])
    rho = rollout_jacobian_spectral_radius(step_fn, x)
    torch.testing.assert_close(rho, torch.tensor(2.0))


def test_spectral_radius_contraction_below_one() -> None:
    """A pure contraction (all |eig| < 1) gives radius < 1."""
    A = torch.diag(torch.tensor([0.3, 0.7, 0.1]))

    def step_fn(x: torch.Tensor) -> torch.Tensor:
        return A @ x

    x = torch.zeros(3)
    rho = rollout_jacobian_spectral_radius(step_fn, x)
    assert rho.item() < 1.0
    torch.testing.assert_close(rho, torch.tensor(0.7))


def test_spectral_radius_expansion_above_one() -> None:
    """An expanding map (some |eig| > 1) gives radius > 1."""
    A = torch.diag(torch.tensor([1.5, 0.2]))

    def step_fn(x: torch.Tensor) -> torch.Tensor:
        return A @ x

    x = torch.zeros(2)
    rho = rollout_jacobian_spectral_radius(step_fn, x)
    assert rho.item() > 1.0
    torch.testing.assert_close(rho, torch.tensor(1.5))


def test_spectral_radius_rotation_scale_complex_eig() -> None:
    """Rotation+scale has complex eigenvalues; radius = the scale factor."""
    scale = 0.9
    theta = 0.5
    A = scale * torch.tensor(
        [
            [torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta))],
            [torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta))],
        ]
    )

    def step_fn(x: torch.Tensor) -> torch.Tensor:
        return A @ x

    x = torch.zeros(2)
    rho = rollout_jacobian_spectral_radius(step_fn, x)
    torch.testing.assert_close(rho, torch.tensor(scale))


def test_spectral_radius_returns_scalar_tensor() -> None:
    """The return value is a 0-dim scalar tensor."""
    A = torch.diag(torch.tensor([0.5, 0.5]))

    def step_fn(x: torch.Tensor) -> torch.Tensor:
        return A @ x

    rho = rollout_jacobian_spectral_radius(step_fn, torch.zeros(2))
    assert isinstance(rho, torch.Tensor)
    assert rho.ndim == 0
