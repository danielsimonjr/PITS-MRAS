"""Phase-1 unit tests for ``pits_mras.utils.hamiltonian`` (IP §4.4).

Covers ``make_skew_symmetric`` (J = -Jᵀ), ``make_positive_definite``
(R = LᵀL + εI ≥ 0, symmetric), ``port_hamiltonian_energy_loss`` (dissipation
residual), and ``hamiltonian_positivity_loss`` (ReLU(-H)).
"""

import torch

from pits_mras.utils.hamiltonian import (
    hamiltonian_positivity_loss,
    make_positive_definite,
    make_skew_symmetric,
    port_hamiltonian_energy_loss,
)


def test_make_skew_symmetric_is_skew() -> None:
    """J == -Jᵀ for arbitrary raw input."""
    raw = torch.randn(4, 3, 3, dtype=torch.float64)
    J = make_skew_symmetric(raw)
    assert J.shape == (4, 3, 3)
    assert torch.allclose(J, -J.transpose(-1, -2), atol=1e-12)
    # Diagonal of a skew matrix is zero.
    diag = torch.diagonal(J, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-12)


def test_make_positive_definite_is_psd_and_symmetric() -> None:
    """R = LᵀL + εI is symmetric with strictly positive eigenvalues."""
    L = torch.randn(5, 3, 3, dtype=torch.float64)
    eps = 1e-6
    R = make_positive_definite(L, epsilon=eps)
    assert R.shape == (5, 3, 3)
    # Symmetric.
    assert torch.allclose(R, R.transpose(-1, -2), atol=1e-10)
    # Positive definite (>= eps because of the εI floor).
    eigvals = torch.linalg.eigvalsh(R)
    assert torch.all(eigvals > 0)
    assert torch.all(eigvals >= eps - 1e-9)


def test_make_positive_definite_zero_L_gives_epsilon_floor() -> None:
    """With L = 0, R = εI exactly."""
    L = torch.zeros(2, 3, 3, dtype=torch.float64)
    eps = 1e-3
    R = make_positive_definite(L, epsilon=eps)
    expected = eps * torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(2, 3, 3)
    assert torch.allclose(R, expected, atol=1e-12)


def test_port_hamiltonian_energy_loss_zero_when_balanced() -> None:
    """Loss is zero when dH/dt = P_control - P_diss exactly."""
    P_control = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    P_diss = torch.tensor([0.5, 0.5, 1.0], dtype=torch.float64)
    dH_dt = P_control - P_diss  # perfectly balanced
    H_pred = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)  # unused in residual
    loss = port_hamiltonian_energy_loss(H_pred, dH_dt, P_control, P_diss)
    assert torch.allclose(loss, torch.zeros((), dtype=torch.float64), atol=1e-12)


def test_port_hamiltonian_energy_loss_positive_when_violated() -> None:
    """Loss equals mean squared residual when the balance is broken."""
    dH_dt = torch.tensor([2.0, 0.0], dtype=torch.float64)
    P_control = torch.tensor([1.0, 0.0], dtype=torch.float64)
    P_diss = torch.tensor([0.0, 0.0], dtype=torch.float64)
    H_pred = torch.tensor([1.0, 1.0], dtype=torch.float64)  # unused in residual
    # residual = dH_dt - P_control + P_diss = [1, 0]; mean(residual²) = 0.5
    loss = port_hamiltonian_energy_loss(H_pred, dH_dt, P_control, P_diss)
    assert torch.allclose(loss, torch.tensor(0.5, dtype=torch.float64), atol=1e-12)


def test_hamiltonian_positivity_loss() -> None:
    """Loss = mean(ReLU(-H)); zero when all H >= 0, positive otherwise."""
    H_pos = torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64)
    assert torch.allclose(
        hamiltonian_positivity_loss(H_pos), torch.zeros((), dtype=torch.float64)
    )
    H_neg = torch.tensor([1.0, -2.0, -4.0], dtype=torch.float64)
    # mean(ReLU(-H)) = mean([0, 2, 4]) = 2.0
    assert torch.allclose(
        hamiltonian_positivity_loss(H_neg), torch.tensor(2.0, dtype=torch.float64)
    )
