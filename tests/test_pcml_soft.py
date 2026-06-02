"""Tests for soft PCML + supporting layers (PCML Addendum §2.2/§2.3).

Covers ``SoftPCMLLoss`` (Patel et al. 2022 augmented loss),
``TaylorNeighborhoodApproximation`` (DAE-HardNet §3 multi-point neighborhood)
and ``LagrangianMultiplierHead`` (KKT warm-start multipliers).
"""

import torch

from pits_mras.constraints import HeatConductionDAE
from pits_mras.models.lagrangian_head import LagrangianMultiplierHead
from pits_mras.models.pcml import SoftPCMLLoss, TaylorNeighborhoodApproximation


# --------------------------------------------------------------------------- #
# SoftPCMLLoss (Patel et al. 2022).
# --------------------------------------------------------------------------- #
def test_soft_pcml_zero_when_constraints_satisfied() -> None:
    """L_pcml_soft = 0 when D=0, h=0 and g<=0 are all satisfied."""
    alpha = 1.0
    dae = HeatConductionDAE(alpha=alpha, T_min=0.0, T_max=100.0)
    loss = SoftPCMLLoss(dae, lambda_diff=1.0, lambda_eq=1.0, lambda_ineq=0.5)
    batch = 8
    x = torch.zeros(batch, 1)
    t = torch.zeros(batch, 1)
    y = torch.full((batch, 1), 25.0)  # within [0, 100]
    d = torch.zeros(batch, 4)
    d[:, 1] = 3.0  # dT_dt
    d[:, 2] = 3.0 / alpha  # d2T_dx2 -> dT_dt - alpha*d2T_dx2 = 0
    total, parts = loss(x, t, y, d)
    assert total.shape == ()
    assert total.item() < 1e-6
    assert parts["diff"].item() < 1e-6


def test_soft_pcml_penalizes_violation() -> None:
    """A larger differential residual yields a larger soft loss."""
    dae = HeatConductionDAE(alpha=1.0, T_min=0.0, T_max=100.0)
    loss = SoftPCMLLoss(dae)
    batch = 4
    x = torch.zeros(batch, 1)
    t = torch.zeros(batch, 1)
    y = torch.full((batch, 1), 25.0)
    d_small = torch.zeros(batch, 4)
    d_small[:, 1] = 0.1  # dT_dt nonzero, d2T_dx2 = 0 -> small residual
    d_big = torch.zeros(batch, 4)
    d_big[:, 1] = 5.0  # larger residual
    total_small, _ = loss(x, t, y, d_small)
    total_big, _ = loss(x, t, y, d_big)
    assert total_big.item() > total_small.item()


def test_soft_pcml_is_differentiable() -> None:
    dae = HeatConductionDAE(alpha=1.0)
    loss = SoftPCMLLoss(dae)
    y = torch.randn(5, 1, requires_grad=True)
    d = torch.randn(5, 4, requires_grad=True)
    total, _ = loss(torch.zeros(5, 1), torch.zeros(5, 1), y, d)
    total.backward()
    assert y.grad is not None and torch.isfinite(y.grad).all()
    assert d.grad is not None and torch.isfinite(d.grad).all()


# --------------------------------------------------------------------------- #
# TaylorNeighborhoodApproximation (DAE-HardNet §3).
# --------------------------------------------------------------------------- #
def test_taylor_approx_converges_to_value_as_delta_small() -> None:
    """As delta -> 0 the neighborhood approximation -> backbone(inputs)."""
    torch.manual_seed(0)
    input_dim, out_dim = 2, 1
    backbone = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 16), torch.nn.Tanh(), torch.nn.Linear(16, out_dim)
    )
    inputs = torch.randn(6, input_dim)
    # Provide the true autograd derivatives so the first-order correction is exact.
    inp = inputs.clone().requires_grad_(True)
    y = backbone(inp)
    derivs = torch.zeros(6, input_dim)
    for i in range(input_dim):
        g = torch.autograd.grad(y.sum(), inp, create_graph=True)[0]
        derivs = g  # [batch, input_dim] (d y / d input_i), out_dim == 1
        break
    approx_layer = TaylorNeighborhoodApproximation(
        backbone, input_dim=input_dim, delta=1e-3, order=1
    )
    y_approx = approx_layer(inputs, derivs)
    assert y_approx.shape == (6, out_dim)
    assert torch.allclose(y_approx, backbone(inputs), atol=1e-2)


# --------------------------------------------------------------------------- #
# LagrangianMultiplierHead (DAE-HardNet warm-start multipliers).
# --------------------------------------------------------------------------- #
def test_lagrangian_head_shapes_and_inequality_nonneg() -> None:
    torch.manual_seed(0)
    head = LagrangianMultiplierHead(
        context_dim=8, n_lambda_eq=3, n_lambda_ineq=2, hidden_dim=16
    )
    ctx = torch.randn(5, 8)
    lam = head(ctx)
    assert lam.shape == (5, 5)  # 3 equality + 2 inequality
    # Inequality multipliers (last 2) must be non-negative (Softplus).
    assert (lam[:, 3:] >= 0).all()


def test_lagrangian_head_all_equality() -> None:
    head = LagrangianMultiplierHead(context_dim=4, n_lambda_eq=2, n_lambda_ineq=0)
    lam = head(torch.randn(3, 4))
    assert lam.shape == (3, 2)
