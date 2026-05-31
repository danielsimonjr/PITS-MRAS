"""Phase-1 unit tests for ``pits_mras.utils.pe_monitor`` (IP §4.5).

Covers the persistence-of-excitation Gram min-eigenvalue check and the
probing-noise helper of ``PEMonitor``.
"""

import torch

from pits_mras.utils.pe_monitor import PEMonitor


def test_pe_not_satisfied_before_window_full() -> None:
    """PE cannot be satisfied until the sliding window is full."""
    mon = PEMonitor(regressor_dim=2, window_size=4, pe_threshold=1e-3)
    for _ in range(3):
        mon.update(torch.randn(2))
    assert mon.is_pe_satisfied() is False


def test_pe_satisfied_for_exciting_signal() -> None:
    """A well-spread regressor history satisfies the PE condition."""
    mon = PEMonitor(regressor_dim=2, window_size=4, pe_threshold=1e-6)
    # Orthogonal directions with large magnitude => Gram min-eig well above 0.
    samples = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
    ]
    for s in samples:
        mon.update(s)
    assert mon.is_pe_satisfied() is True


def test_pe_not_satisfied_for_collinear_signal() -> None:
    """A rank-deficient (collinear) history fails the PE condition."""
    mon = PEMonitor(regressor_dim=2, window_size=4, pe_threshold=1e-3)
    for _ in range(4):
        mon.update(torch.tensor([1.0, 1.0]))  # all along one direction
    # Gram matrix is rank-1 => min eigenvalue ~ 0 < threshold.
    assert mon.is_pe_satisfied() is False


def test_probing_noise_shape_and_scale() -> None:
    """Probing noise has the requested dimension and ~noise_std scale."""
    mon = PEMonitor(regressor_dim=2, window_size=4, noise_std=0.05)
    noise = mon.get_probing_noise(control_dim=3)
    assert noise.shape == (3,)
    # Magnitude is finite and on the order of noise_std (loose bound).
    assert torch.isfinite(noise).all()
    assert noise.abs().max() < 1.0
