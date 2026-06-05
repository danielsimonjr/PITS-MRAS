"""Equivalence tests for the vectorized KKT constraint Jacobian (ROADMAP #5).

The :class:`KKTProjectionLayer` builds the constraint Jacobians ``Jc`` (the
differential + equality block, w.r.t. ``[y, d]``) and ``Jg`` (the inequality
block, w.r.t. ``y``) inside ``_constraints_and_jac``. The original
implementation looped per constraint row with ``torch.autograd.grad``; the
refactor vectorizes this with ``torch.func`` (``jacrev`` + ``vmap``).

These tests pin the *behaviour*: the vectorized Jacobians must equal the
reference per-row autograd loop to floating-point tolerance, and the projection
forward output + the implicit-function-theorem gradient w.r.t. ``y_hat`` must be
unchanged.
"""

import torch

from pits_mras.constraints import HeatConductionDAE, MechanicalDAE
from pits_mras.constraints.base import ConstraintSpec, PhysicsConstraints
from pits_mras.models.pcml import KKTProjectionLayer


class _SumEqualsOneDAE(PhysicsConstraints):
    """Single linear equality ``h(y) = sum(y) - 1 = 0`` (mirrors test_pcml_hard)."""

    def __init__(self, n: int = 3) -> None:
        self.n = n
        self._spec = ConstraintSpec(n_differential=0, n_equality=1, n_inequality=0, n_outputs=n)

    @property
    def spec(self) -> ConstraintSpec:
        return self._spec

    def differential(self, x, t, y, d):
        return torch.zeros(y.shape[0], 0, device=y.device, dtype=y.dtype)

    def equality(self, x, t, y):
        return y.sum(dim=-1, keepdim=True) - 1.0

    def inequality(self, x, t, y):
        return torch.zeros(y.shape[0], 0, device=y.device, dtype=y.dtype)


def _reference_constraints_and_jac(layer, x, t, z):
    """Reference (old, per-row autograd loop) build of ``(c, g, Jc_y, Jc_d, Jg_y)``.

    This is a faithful copy of the pre-refactor loop, kept here so the
    equivalence test compares the vectorized layer against the original
    behaviour without depending on git history.
    """
    zc = z.detach().requires_grad_(True)
    y = zc[:, : layer.n_y]
    d = zc[:, layer.n_y : layer.n_y + layer.n_d]
    parts = []
    if layer.n_diff > 0:
        parts.append(layer.constraints.differential(x, t, y, d))
    if layer.n_eq > 0:
        parts.append(layer.constraints.equality(x, t, y))
    c = torch.cat(parts, dim=-1) if parts else z.new_zeros(z.shape[0], 0)
    g = layer.constraints.inequality(x, t, y) if layer.n_g > 0 else z.new_zeros(z.shape[0], 0)
    b = z.shape[0]
    jc = z.new_zeros(b, layer.n_c, layer.n_y + layer.n_d)
    for k in range(layer.n_c):
        grad_k = torch.autograd.grad(c[:, k].sum(), zc, retain_graph=True)[0]
        jc[:, k, :] = grad_k[:, : layer.n_y + layer.n_d]
    jg = z.new_zeros(b, layer.n_g, layer.n_y)
    for k in range(layer.n_g):
        grad_k = torch.autograd.grad(g[:, k].sum(), zc, retain_graph=True)[0]
        jg[:, k, :] = grad_k[:, : layer.n_y]
    jc_y = jc[:, :, : layer.n_y]
    jc_d = jc[:, :, layer.n_y : layer.n_y + layer.n_d]
    return c.detach(), g.detach(), jc_y.detach(), jc_d.detach(), jg.detach()


def _make_heat_layer():
    dae = HeatConductionDAE(alpha=0.8, T_min=-50.0, T_max=50.0)
    return KKTProjectionLayer(dae, n_output=1, n_deriv=4, max_newton_iter=25), dae


def _make_holonomic_layer():
    n, m = 2, 1

    def inertia(q):
        return torch.eye(n).unsqueeze(0).expand(q.shape[0], n, n)

    def zero_force(*args):
        return torch.zeros(args[0].shape[0], n)

    def jacobian(q):
        return torch.tensor([[1.0, -1.0]]).unsqueeze(0).expand(q.shape[0], m, n)

    dae = MechanicalDAE(
        n_joints=n,
        n_holonomic=m,
        inertia_fn=inertia,
        coriolis_fn=zero_force,
        gravity_fn=zero_force,
        actuator_fn=lambda q: torch.zeros(q.shape[0], n, m),
        constraint_fn=jacobian,
        q_bounds=(torch.full((n,), -2.0), torch.full((n,), 2.0)),
    )
    layer = KKTProjectionLayer(dae, n_output=2 * n, n_deriv=2 * n + m, max_newton_iter=30)
    return layer, dae


def _random_z(layer, batch, seed):
    torch.manual_seed(seed)
    return torch.randn(batch, layer.N)


def test_vectorized_jac_matches_reference_loop_heat() -> None:
    """Heat DAE: differential (depends on d) + 2 inequalities (depend on y)."""
    layer, _ = _make_heat_layer()
    x = torch.zeros(5, 1)
    t = torch.zeros(5, 1)
    z = _random_z(layer, 5, seed=7)
    c, g, jc_y, jc_d, jg = layer._constraints_and_jac(x, t, z)
    rc, rg, rjc_y, rjc_d, rjg = _reference_constraints_and_jac(layer, x, t, z)
    torch.testing.assert_close(c, rc, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(g, rg, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(jc_y, rjc_y, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(jc_d, rjc_d, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(jg, rjg, rtol=1e-5, atol=1e-7)


def test_vectorized_jac_matches_reference_loop_equality() -> None:
    """Linear-equality DAE: equality block only, no inequalities, no d."""
    dae = _SumEqualsOneDAE(n=3)
    layer = KKTProjectionLayer(dae, n_output=3, n_deriv=0, max_newton_iter=20)
    x = torch.zeros(4, 1)
    t = torch.zeros(4, 1)
    z = _random_z(layer, 4, seed=11)
    c, g, jc_y, jc_d, jg = layer._constraints_and_jac(x, t, z)
    rc, rg, rjc_y, rjc_d, rjg = _reference_constraints_and_jac(layer, x, t, z)
    torch.testing.assert_close(c, rc, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(jc_y, rjc_y, rtol=1e-5, atol=1e-7)
    assert jc_d.shape == rjc_d.shape  # n_d == 0
    assert jg.shape == rjg.shape  # n_g == 0


def test_vectorized_jac_matches_reference_loop_holonomic() -> None:
    """Holonomic MechanicalDAE: differential (depends on y and d, with J^T lambda)
    plus joint-limit inequalities (depend on y)."""
    layer, _ = _make_holonomic_layer()
    x = torch.zeros(4, 1)
    t = torch.zeros(4, 1)
    z = _random_z(layer, 4, seed=13)
    c, g, jc_y, jc_d, jg = layer._constraints_and_jac(x, t, z)
    rc, rg, rjc_y, rjc_d, rjg = _reference_constraints_and_jac(layer, x, t, z)
    torch.testing.assert_close(c, rc, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(g, rg, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(jc_y, rjc_y, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(jc_d, rjc_d, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(jg, rjg, rtol=1e-5, atol=1e-7)


def test_projection_forward_and_ift_gradient_unchanged_heat() -> None:
    """Golden: forward output and IFT gradient w.r.t. y_hat are numerically
    unchanged by the vectorized Jacobian (values captured from the original
    per-row-loop implementation)."""
    layer, dae = _make_heat_layer()
    torch.manual_seed(1)
    x = torch.zeros(5, 1)
    t = torch.zeros(5, 1)
    y_hat = (torch.randn(5, 1) * 5.0).requires_grad_(True)
    d_hat = torch.randn(5, 4)
    lam_hat = torch.zeros(5, dae.spec.n_differential + dae.spec.n_inequality)

    y_t, d_t, _ = layer(x, t, y_hat, d_hat, lam_hat)
    grad = torch.autograd.grad(y_t.pow(2).sum(), y_hat)[0]

    # Golden values from the original loop-based code (same seed / construction).
    golden_y = torch.tensor(
        [
            [3.306760787963867],
            [1.3346205949783325],
            [0.30838629603385925],
            [3.1065866947174072],
            [-2.2595298290252686],
        ]
    )
    golden_grad = torch.tensor(
        [
            [6.613521575927734],
            [2.669241189956665],
            [0.6167725920677185],
            [6.2131733894348145],
            [-4.519059658050537],
        ]
    )
    torch.testing.assert_close(y_t.detach(), golden_y, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grad, golden_grad, rtol=1e-5, atol=1e-6)


def test_projection_forward_matches_reference_loop_holonomic() -> None:
    """End-to-end: a layer using the vectorized Jacobian and an otherwise
    identical layer using the reference loop produce the same projection."""
    layer, dae = _make_holonomic_layer()
    ref_layer, _ = _make_holonomic_layer()
    # Swap the reference layer's Jacobian builder for the per-row loop.
    ref_layer._constraints_and_jac = (  # type: ignore[method-assign]
        lambda x, t, z: _reference_constraints_and_jac(ref_layer, x, t, z)
    )

    torch.manual_seed(5)
    b = 4
    x = torch.zeros(b, 1)
    t = torch.zeros(b, 1)
    y_hat = torch.randn(b, layer.n_y, requires_grad=True)
    y_hat_ref = y_hat.detach().clone().requires_grad_(True)
    d_hat = torch.randn(b, layer.n_d)
    lam_hat = torch.zeros(b, dae.spec.n_differential + dae.spec.n_inequality)

    y_t, d_t, lam_t = layer(x, t, y_hat, d_hat, lam_hat)
    ry_t, rd_t, rlam_t = ref_layer(x, t, y_hat_ref, d_hat, lam_hat)
    torch.testing.assert_close(y_t, ry_t, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(d_t, rd_t, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(lam_t, rlam_t, rtol=1e-5, atol=1e-6)

    g = torch.autograd.grad(y_t.pow(2).sum(), y_hat)[0]
    rg = torch.autograd.grad(ry_t.pow(2).sum(), y_hat_ref)[0]
    torch.testing.assert_close(g, rg, rtol=1e-5, atol=1e-6)
