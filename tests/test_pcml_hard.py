"""Tests for hard PCML: the KKT projection layer + PCMLModule (DAE-HardNet).

The projection solves ``min ½||y-ŷ||² s.t. D=0, h=0, g<=0`` via a differentiable
KKT-Newton solve. We verify it against a closed-form linear-equality projection,
that it drives the heat-equation violation to ~0, that gradients flow, and that
the PCMLModule switches soft->hard at the eta threshold.
"""

import torch

from pits_mras.constraints import HeatConductionDAE, MechanicalDAE
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


def test_kkt_projection_reports_convergence() -> None:
    """The projection surfaces a convergence signal (no longer silently returns
    a non-stationary iterate when Newton exhausts max_iter)."""
    dae = HeatConductionDAE(alpha=0.8, T_min=-50.0, T_max=50.0)
    torch.manual_seed(0)
    x = torch.zeros(4, 1)
    t = torch.zeros(4, 1)
    y_hat = torch.randn(4, 1)
    d_hat = torch.randn(4, 4)
    lam_hat = torch.zeros(4, 3)

    ok = KKTProjectionLayer(dae, n_output=1, n_deriv=4, max_newton_iter=30)
    ok(x, t, y_hat, d_hat, lam_hat)
    assert ok.last_converged is True
    assert ok.last_residual < ok.newton_tol

    # With no Newton iterations the projection reports the (high) initial
    # residual as non-converged. (This near-affine heat-eq projection converges
    # in a single Newton step, so 0 iterations is needed to show non-convergence;
    # last_residual now reflects the RETURNED iterate, not the pre-step one.)
    bad = KKTProjectionLayer(dae, n_output=1, n_deriv=4, max_newton_iter=0)
    bad(x, t, y_hat, d_hat, lam_hat)
    assert bad.last_converged is False
    assert bad.last_residual >= bad.newton_tol


class _SteepEqDAE(PhysicsConstraints):
    """Single high-curvature equality ``h(y) = atan(k*y) = 0`` (root y=0).

    For large ``k`` the curvature makes an undamped full Newton step overshoot and
    diverge from a far start -- the stiff case for the line-search test.
    """

    def __init__(self, k: float = 8.0) -> None:
        self.k = k
        self._spec = ConstraintSpec(
            n_differential=0, n_equality=1, n_inequality=0, n_outputs=1
        )

    @property
    def spec(self) -> ConstraintSpec:
        return self._spec

    def differential(self, x, t, y, d):
        return torch.zeros(y.shape[0], 0, device=y.device, dtype=y.dtype)

    def equality(self, x, t, y):
        return torch.atan(self.k * y)  # [batch, 1]

    def inequality(self, x, t, y):
        return torch.zeros(y.shape[0], 0, device=y.device, dtype=y.dtype)


def test_kkt_line_search_keeps_residual_bounded_on_stiff_projection() -> None:
    """Line search keeps the Newton iterate BOUNDED on a stiff high-curvature
    projection where the undamped full step diverges (carried-forward gap #1).

    The undamped step overshoots and the residual blows up by orders of
    magnitude; the line search guarantees a non-increasing residual, so it stays
    bounded (a far more usable iterate for the implicit-function gradient). (The
    atan constraint's gradient vanishes far from the root, so line search bounds
    rather than fully converges it — the honest, demonstrable improvement.)
    """
    dae = _SteepEqDAE(k=8.0)
    x = torch.zeros(3, 1)
    t = torch.zeros(3, 1)
    y_hat = torch.tensor([[2.0], [3.0], [5.0]])  # far from the root
    d_hat = torch.zeros(3, 0)
    lam_hat = torch.zeros(3, 1)
    # Initial L-inf residual of the projection system (bounded; atan <= pi/2).
    init_res = float(torch.atan(8.0 * y_hat).abs().max())

    ls = KKTProjectionLayer(
        dae, n_output=1, n_deriv=0, max_newton_iter=40, use_line_search=True
    )
    ls(x, t, y_hat, d_hat, lam_hat)
    no = KKTProjectionLayer(
        dae, n_output=1, n_deriv=0, max_newton_iter=40, use_line_search=False
    )
    no(x, t, y_hat, d_hat, lam_hat)

    # Undamped diverges (residual explodes); line search stays bounded.
    assert no.last_residual > 1e3
    assert ls.last_residual <= init_res + 1e-6  # non-increasing -> bounded
    assert ls.last_residual < no.last_residual / 1e3


def test_kkt_line_search_matches_full_step_on_affine() -> None:
    """On a well-behaved affine constraint the line search is a no-op: it accepts
    the full step, so the output matches use_line_search=False exactly."""
    torch.manual_seed(0)
    n = 3
    x = torch.zeros(4, 1)
    t = torch.zeros(4, 1)
    y_hat = torch.randn(4, n)
    d_hat = torch.zeros(4, 0)
    lam_hat = torch.zeros(4, 1)

    proj_ls = KKTProjectionLayer(
        _SumEqualsOneDAE(n=n), n_output=n, n_deriv=0, max_newton_iter=20,
        use_line_search=True,
    )
    proj_no = KKTProjectionLayer(
        _SumEqualsOneDAE(n=n), n_output=n, n_deriv=0, max_newton_iter=20,
        use_line_search=False,
    )
    y_ls, _, _ = proj_ls(x, t, y_hat, d_hat, lam_hat)
    y_no, _, _ = proj_no(x, t, y_hat, d_hat, lam_hat)
    assert torch.allclose(y_ls, y_no, atol=1e-6)


def test_kkt_projection_on_holonomic_mechanical_dae() -> None:
    """The KKT projection is well-formed and reduces violation on a holonomic
    MechanicalDAE (regression for the spec-width fix that makes n_c match the
    EOM + Psi + J q_dot residual width)."""
    n, m = 2, 1

    def inertia(q):
        return torch.eye(n).unsqueeze(0).expand(q.shape[0], n, n)

    def zero_force(*args):
        return torch.zeros(args[0].shape[0], n)

    def jacobian(q):
        return torch.tensor([[1.0, -1.0]]).unsqueeze(0).expand(q.shape[0], m, n)

    dae = MechanicalDAE(
        n_joints=n, n_holonomic=m, inertia_fn=inertia, coriolis_fn=zero_force,
        gravity_fn=zero_force, actuator_fn=lambda q: torch.zeros(q.shape[0], n, m),
        constraint_fn=jacobian,
    )
    # y = [q, q_dot] (2n), d = [q_dot, q_ddot, lambda] (2n + m).
    n_out, n_der = 2 * n, 2 * n + m
    proj = KKTProjectionLayer(dae, n_output=n_out, n_deriv=n_der, max_newton_iter=30)
    b = 4
    x = torch.zeros(b, 1)
    t = torch.zeros(b, 1)
    y_hat = torch.randn(b, n_out)
    d_hat = torch.randn(b, n_der)
    lam_hat = torch.zeros(b, dae.spec.n_differential + dae.spec.n_inequality)

    v_before = dae.violation(x, t, y_hat, d_hat)
    y_t, d_t, _ = proj(x, t, y_hat, d_hat, lam_hat)
    v_after = dae.violation(x, t, y_t, d_t)
    assert y_t.shape == (b, n_out) and d_t.shape == (b, n_der)
    assert v_after.item() < v_before.item()
    assert v_after.item() < 1e-3


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
