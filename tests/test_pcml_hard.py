"""Tests for hard PCML: the KKT projection layer + PCMLModule (DAE-HardNet).

The projection solves ``min ½||y-ŷ||² s.t. D=0, h=0, g<=0`` via a differentiable
KKT-Newton solve. We verify it against a closed-form linear-equality projection,
that it drives the heat-equation violation to ~0, that gradients flow, and that
the PCMLModule switches soft->hard at the eta threshold.
"""

import torch

from pits_mras.constraints import HeatConductionDAE
from pits_mras.constraints.base import ConstraintSpec, PhysicsConstraints
from pits_mras.models.pcml import KKTProjectionLayer, PCMLModule


class _SumEqualsOneDAE(PhysicsConstraints):
    """Toy: a single linear equality h(y) = sum(y) - 1 = 0; no diff/inequality.

    The Euclidean projection of ŷ onto {y : 1ᵀy = 1} is closed-form:
    ỹ = ŷ - ((1ᵀŷ - 1)/n) · 1, which we use to validate the KKT solve.
    """

    def __init__(self, n: int = 2) -> None:
        self.n = n
        self._spec = ConstraintSpec(
            n_differential=0, n_equality=1, n_inequality=0, n_outputs=n
        )

    @property
    def spec(self) -> ConstraintSpec:
        return self._spec

    def differential(self, x, t, y, d):
        return torch.zeros(y.shape[0], 0, device=y.device, dtype=y.dtype)

    def equality(self, x, t, y):
        return y.sum(dim=-1, keepdim=True) - 1.0  # [batch, 1]

    def inequality(self, x, t, y):
        return torch.zeros(y.shape[0], 0, device=y.device, dtype=y.dtype)


def test_kkt_projection_matches_closed_form_linear_equality() -> None:
    torch.manual_seed(0)
    n = 3
    dae = _SumEqualsOneDAE(n=n)
    proj = KKTProjectionLayer(dae, n_output=n, n_deriv=0, max_newton_iter=20)
    batch = 4
    x = torch.zeros(batch, 1)
    t = torch.zeros(batch, 1)
    y_hat = torch.randn(batch, n)
    d_hat = torch.zeros(batch, 0)
    lam_hat = torch.zeros(batch, 1)  # one equality multiplier

    y_tilde, _, _ = proj(x, t, y_hat, d_hat, lam_hat)
    # Closed-form projection.
    expected = y_hat - ((y_hat.sum(-1, keepdim=True) - 1.0) / n)
    assert torch.allclose(y_tilde, expected, atol=1e-4)
    # And the equality is satisfied.
    assert torch.allclose(y_tilde.sum(-1), torch.ones(batch), atol=1e-4)


def test_kkt_projection_reduces_heat_violation() -> None:
    torch.manual_seed(1)
    alpha = 0.8
    dae = HeatConductionDAE(alpha=alpha, T_min=-50.0, T_max=50.0)
    proj = KKTProjectionLayer(dae, n_output=1, n_deriv=4, max_newton_iter=25)
    batch = 5
    x = torch.zeros(batch, 1)
    t = torch.zeros(batch, 1)
    y_hat = torch.randn(batch, 1) * 5.0  # within temperature bounds
    d_hat = torch.randn(batch, 4)  # generic derivatives -> heat eq violated
    lam_hat = torch.zeros(batch, dae.spec.n_differential + dae.spec.n_inequality)

    v_before = dae.violation(x, t, y_hat, d_hat)
    y_t, d_t, _ = proj(x, t, y_hat, d_hat, lam_hat)
    v_after = dae.violation(x, t, y_t, d_t)
    assert v_after.item() < 1e-4
    assert v_after.item() < v_before.item()


def test_kkt_projection_is_differentiable() -> None:
    torch.manual_seed(2)
    n = 2
    dae = _SumEqualsOneDAE(n=n)
    proj = KKTProjectionLayer(dae, n_output=n, n_deriv=0, max_newton_iter=15)
    y_hat = torch.randn(3, n, requires_grad=True)
    d_hat = torch.zeros(3, 0)
    lam_hat = torch.zeros(3, 1)
    y_t, _, _ = proj(torch.zeros(3, 1), torch.zeros(3, 1), y_hat, d_hat, lam_hat)
    y_t.pow(2).sum().backward()
    assert y_hat.grad is not None and torch.isfinite(y_hat.grad).all()
    assert y_hat.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# PCMLModule dynamic activation (soft -> hard at eta).
# --------------------------------------------------------------------------- #
def test_pcml_module_dynamic_activation() -> None:
    dae = HeatConductionDAE(alpha=1.0)
    backbone = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    mod = PCMLModule(
        constraints=dae, backbone=backbone, input_dim=2,
        n_output=1, n_deriv=4, n_lambda=dae.spec.n_differential + dae.spec.n_inequality,
        eta=0.01,
    )
    assert mod.mode == "soft"
    # A data loss above eta does not activate hard mode.
    assert mod.update_activation(0.5) is False
    assert mod.mode == "soft"
    # Dropping below eta flips it on (once).
    assert mod.update_activation(0.001) is True
    assert mod.mode == "hard"
    assert mod.update_activation(0.0005) is False  # already active


def test_pcml_module_soft_forward_runs() -> None:
    dae = HeatConductionDAE(alpha=1.0, T_min=-100.0, T_max=100.0)
    backbone = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    n_lambda = dae.spec.n_differential + dae.spec.n_inequality
    mod = PCMLModule(
        constraints=dae, backbone=backbone, input_dim=2,
        n_output=1, n_deriv=4, n_lambda=n_lambda, eta=0.01,
    )
    x = torch.zeros(4, 1)
    t = torch.zeros(4, 1)
    y_hat = torch.randn(4, 1)
    d_hat = torch.randn(4, 4)
    lam_hat = torch.zeros(4, n_lambda)
    y_out, loss, info = mod(x, t, y_hat, d_hat, lam_hat, y_true=torch.randn(4, 1))
    assert y_out.shape == (4, 1)
    assert info["mode"] == "soft"
    assert torch.isfinite(loss)
