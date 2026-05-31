"""Phase-3 loss tests with real numerical assertions.

Covers physics energy balance, temporal multi-step prediction, stability
ReLU behaviour, the HJB residual at the optimum, and TotalLoss aggregation.
The IRL load-bearing checks live in ``tests/test_irl.py``.
"""
from __future__ import annotations

import math

import torch

from pits_mras.losses import TotalLoss
from pits_mras.losses.hjb import HJBResidualLoss, LyapunovDecreaseEnforcer
from pits_mras.losses.physics import PhysicsLoss
from pits_mras.losses.stability import (
    ControlEffortLoss,
    LyapunovConstraintLoss,
    MRASStabilityLoss,
    ParameterBoundednessLoss,
)
from pits_mras.losses.temporal import (
    AttentionRegularizationLoss,
    MultiStepPredictionLoss,
    TemporalLoss,
    TemporalSmoothnessLoss,
)
from pits_mras.models.critic import QuadraticCritic


def _inject_P(critic: QuadraticCritic, P: torch.Tensor) -> None:
    """Set W_c so V̂(e) = eᵀ P e (diag coeff P[i,i], off-diag 2 P[i,j])."""
    n = critic.state_dim
    coeffs = []
    for i in range(n):
        for j in range(i, n):
            coeffs.append(P[i, j] if i == j else P[i, j] + P[j, i])
    w = torch.stack(coeffs).to(critic.W_c.weight.dtype)
    with torch.no_grad():
        critic.W_c.weight.copy_(w.unsqueeze(0))


# --------------------------------------------------------------------------- #
# Physics
# --------------------------------------------------------------------------- #
class TestPhysics:
    def test_energy_balance_zero_when_consistent(self) -> None:
        loss = PhysicsLoss()
        p_ctrl = torch.tensor([1.0, 2.0, -3.0])
        p_diss = torch.tensor([0.5, 0.5, 0.5])
        dh_dt = p_ctrl - p_diss  # exact energy balance -> residual 0
        out = loss(dh_dt, p_ctrl, p_diss)
        assert out["energy"].item() < 1e-12
        assert out["loss"].item() < 1e-12

    def test_optional_residuals_default_zero(self) -> None:
        loss = PhysicsLoss()
        dh_dt = torch.zeros(3)
        out = loss(dh_dt, torch.ones(3), torch.ones(3))
        assert out["pde"].item() == 0.0
        assert out["bc"].item() == 0.0
        assert out["sym"].item() == 0.0

    def test_pde_residual_contributes(self) -> None:
        loss = PhysicsLoss(lambda_energy=0.0, lambda_pde=2.0)
        dh_dt = torch.zeros(2)
        pde = torch.tensor([1.0, 1.0])
        out = loss(dh_dt, torch.zeros(2), torch.zeros(2), pde_residual=pde)
        assert math.isclose(out["loss"].item(), 2.0 * 1.0, rel_tol=1e-6)

    def test_differentiable(self) -> None:
        loss = PhysicsLoss()
        dh_dt = torch.zeros(3, requires_grad=True)
        out = loss(dh_dt, torch.ones(3), torch.zeros(3))
        out["loss"].backward()
        assert dh_dt.grad is not None


# --------------------------------------------------------------------------- #
# Temporal
# --------------------------------------------------------------------------- #
class TestTemporal:
    def test_zero_when_perfect_prediction(self) -> None:
        loss = TemporalLoss(horizon=4)
        targets = torch.randn(3, 4, 2)
        out = loss(targets.clone(), targets.clone())
        assert out["loss"].item() < 1e-12

    def test_attention_weighting(self) -> None:
        loss = TemporalLoss(horizon=2)
        pred = torch.zeros(1, 2, 1)
        tgt = torch.tensor([[[1.0], [3.0]]])  # sq err = [1, 9]
        attn = torch.tensor([[1.0, 0.0]])     # weight only first step
        out = loss(pred, tgt, attention_weights=attn)
        assert math.isclose(out["loss"].item(), 1.0, rel_tol=1e-6)

    def test_smoothness_penalty(self) -> None:
        loss = TemporalLoss(horizon=3, lambda_smooth=1.0)
        pred = torch.tensor([[[0.0], [1.0], [3.0]]])  # diffs 1,2 -> sq [1,4] mean 2.5
        out = loss(pred, pred)  # prediction term 0
        assert math.isclose(out["smoothness"].item(), 2.5, rel_tol=1e-6)
        assert math.isclose(out["loss"].item(), 2.5, rel_tol=1e-6)

    def test_multistep_component(self) -> None:
        loss = MultiStepPredictionLoss()
        pred = torch.zeros(2, 3, 2)
        tgt = torch.ones(2, 3, 2)
        out = loss(pred, tgt)
        # per-step sq err = 2, mean over horizon = 2, mean over batch = 2
        assert math.isclose(out.item(), 2.0, rel_tol=1e-6)

    def test_attention_regularization_entropy(self) -> None:
        reg = AttentionRegularizationLoss()
        uniform = torch.full((1, 4), 0.25)
        peaked = torch.tensor([[0.97, 0.01, 0.01, 0.01]])
        # uniform has max entropy -> smallest negative-entropy penalty
        assert reg(uniform).item() < reg(peaked).item()

    def test_smoothness_class(self) -> None:
        smooth = TemporalSmoothnessLoss()
        pred = torch.tensor([[[0.0], [2.0]]])  # diff 2 -> 4
        assert math.isclose(smooth(pred).item(), 4.0, rel_tol=1e-6)

    def test_differentiable(self) -> None:
        loss = TemporalLoss(horizon=2, lambda_smooth=0.5)
        pred = torch.randn(2, 2, 2, requires_grad=True)
        tgt = torch.randn(2, 2, 2)
        loss(pred, tgt)["loss"].backward()
        assert pred.grad is not None


# --------------------------------------------------------------------------- #
# Stability
# --------------------------------------------------------------------------- #
class TestStability:
    def test_lyapunov_no_penalty_when_decreasing(self) -> None:
        loss = LyapunovConstraintLoss(margin=0.0)
        vdot = torch.tensor([-1.0, -2.0, -0.5])  # all decreasing
        assert loss(vdot).item() == 0.0

    def test_lyapunov_penalizes_increase(self) -> None:
        loss = LyapunovConstraintLoss(margin=0.0)
        vdot = torch.tensor([1.0, -1.0, 2.0])  # relu -> [1,0,2], mean = 1
        assert math.isclose(loss(vdot).item(), 1.0, rel_tol=1e-6)

    def test_lyapunov_margin(self) -> None:
        loss = LyapunovConstraintLoss(margin=0.5)
        vdot = torch.tensor([-0.2])  # relu(-0.2 + 0.5) = 0.3
        assert math.isclose(loss(vdot).item(), 0.3, rel_tol=1e-6)

    def test_parameter_boundedness_l2(self) -> None:
        loss = ParameterBoundednessLoss()
        params = [torch.tensor([3.0, 4.0])]  # ||.||^2 = 25
        assert math.isclose(loss(params).item(), 25.0, rel_tol=1e-6)

    def test_control_effort(self) -> None:
        R = torch.eye(2)
        loss = ControlEffortLoss(R)
        u = torch.tensor([[3.0, 4.0]])  # uᵀRu = 25
        assert math.isclose(loss(u).item(), 25.0, rel_tol=1e-6)

    def test_mras_aggregator(self) -> None:
        R = torch.eye(1)
        agg = MRASStabilityLoss(R)
        vdot = torch.tensor([1.0])
        u = torch.tensor([[2.0]])
        params = [torch.tensor([1.0])]
        out = agg(vdot, u, params)
        assert "loss" in out and "lyapunov" in out
        assert out["loss"].item() > 0.0

    def test_differentiable(self) -> None:
        loss = LyapunovConstraintLoss()
        vdot = torch.tensor([1.0, 2.0], requires_grad=True)
        loss(vdot).backward()
        assert vdot.grad is not None


# --------------------------------------------------------------------------- #
# HJB
# --------------------------------------------------------------------------- #
class TestHJB:
    def test_residual_zero_at_optimum_1d(self) -> None:
        # 1D system ė = a e + b u, cost ∫(q e² + r u²).  With V̂ = P e² and the
        # ½-scaled optimal control u* = −R⁻¹ b P e = −K e, the CARE is
        #   q + 2 a P − b² P² / r = 0  =>  P = (a + √(a² + b² q / r)) · r / b².
        a, b, q, r = 0.5, 1.0, 1.0, 1.0
        P = (a + math.sqrt(a * a + b * b * q / r)) * r / (b * b)
        A = torch.tensor([[a]], dtype=torch.float64)
        B = torch.tensor([[b]], dtype=torch.float64)
        Q = torch.tensor([[q]], dtype=torch.float64)
        R = torch.tensor([[r]], dtype=torch.float64)

        critic = QuadraticCritic(1).double()
        _inject_P(critic, torch.tensor([[P]], dtype=torch.float64))

        loss = HJBResidualLoss(A, B, Q, R)
        e = torch.tensor([[1.0], [-2.0], [0.3]], dtype=torch.float64)
        out = loss(critic, e)
        assert out["loss"].item() < 1e-9
        # u* should equal the LQR gain -K e with K = R⁻¹ b P.
        K = b * P / r
        assert torch.allclose(out["u_star"], -K * e, atol=1e-9)

    def test_residual_nonzero_off_optimum(self) -> None:
        a, b, q, r = 0.5, 1.0, 1.0, 1.0
        A = torch.tensor([[a]], dtype=torch.float64)
        B = torch.tensor([[b]], dtype=torch.float64)
        Q = torch.tensor([[q]], dtype=torch.float64)
        R = torch.tensor([[r]], dtype=torch.float64)
        critic = QuadraticCritic(1).double()  # default (wrong) P
        loss = HJBResidualLoss(A, B, Q, R)
        e = torch.tensor([[1.0], [-2.0]], dtype=torch.float64)
        assert loss(critic, e)["loss"].item() > 1e-3

    def test_lyapunov_decrease_enforcer(self) -> None:
        enforcer = LyapunovDecreaseEnforcer(margin=0.0)
        grad_v = torch.tensor([[1.0, 0.0]])
        f = torch.tensor([[2.0, 5.0]])  # grad.f = 2 > 0 -> penalty 2
        assert math.isclose(enforcer(grad_v, f).item(), 2.0, rel_tol=1e-6)
        f2 = torch.tensor([[-2.0, 5.0]])  # grad.f = -2 -> no penalty
        assert enforcer(grad_v, f2).item() == 0.0

    def test_differentiable(self) -> None:
        A = torch.tensor([[0.5]])
        B = torch.tensor([[1.0]])
        Q = torch.tensor([[1.0]])
        R = torch.tensor([[1.0]])
        critic = QuadraticCritic(1)
        loss = HJBResidualLoss(A, B, Q, R)
        e = torch.tensor([[1.0]], requires_grad=True)
        loss(critic, e)["loss"].backward()
        assert critic.W_c.weight.grad is not None


# --------------------------------------------------------------------------- #
# TotalLoss
# --------------------------------------------------------------------------- #
class TestTotalLoss:
    def test_aggregates_components(self) -> None:
        total = TotalLoss()
        components = {
            "physics": torch.tensor(1.0),
            "temporal": torch.tensor(2.0),
            "stability": torch.tensor(3.0),
            "irl": torch.tensor(4.0),
            "hjb": torch.tensor(5.0),
            "costate": torch.tensor(6.0),
            "data": torch.tensor(7.0),
        }
        out = total(components)
        # LossConfig defaults: physics=1.0, temporal=0.5, stability=2.0,
        # irl=1.0, hjb=0.01, costate=0.1, data=1.0.
        expected = (
            1.0 * 1 + 0.5 * 2 + 2.0 * 3 + 1.0 * 4
            + 0.01 * 5 + 0.1 * 6 + 1.0 * 7
        )
        assert math.isclose(out["loss"].item(), expected, rel_tol=1e-6)
        for key in ("loss/physics", "loss/temporal", "loss/hjb", "loss/data"):
            assert key in out

    def test_missing_components_treated_as_zero(self) -> None:
        total = TotalLoss()
        out = total({"physics": torch.tensor(2.0)})
        assert math.isclose(out["loss"].item(), 2.0, rel_tol=1e-6)

    def test_differentiable(self) -> None:
        total = TotalLoss()
        x = torch.tensor(1.0, requires_grad=True)
        out = total({"physics": x * 3})
        out["loss"].backward()
        assert x.grad is not None
