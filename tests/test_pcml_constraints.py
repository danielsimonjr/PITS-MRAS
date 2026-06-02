"""Tests for the PCML constraints library (PCML Addendum §2.1).

Targets ``pits_mras.constraints`` (``PhysicsConstraints`` ABC + ``ConstraintSpec``,
``MechanicalDAE``, ``HeatConductionDAE``). These constraint systems define the
DAE residuals (differential ``D``, equality ``h``, inequality ``g``) that feed
both the soft PCML loss and the hard KKT projection.
"""

import torch

from pits_mras.constraints import (
    ConstraintSpec,
    HeatConductionDAE,
    MechanicalDAE,
    PhysicsConstraints,
)


# --------------------------------------------------------------------------- #
# ConstraintSpec + ABC contract.
# --------------------------------------------------------------------------- #
def test_constraint_spec_defaults() -> None:
    spec = ConstraintSpec()
    assert (spec.n_differential, spec.n_equality, spec.n_inequality, spec.n_outputs) == (
        0,
        0,
        0,
        0,
    )
    spec2 = ConstraintSpec(n_differential=2, n_equality=1, n_inequality=4, n_outputs=2)
    assert spec2.n_differential == 2 and spec2.n_inequality == 4


def test_dae_classes_are_physics_constraints() -> None:
    assert issubclass(MechanicalDAE, PhysicsConstraints)
    assert issubclass(HeatConductionDAE, PhysicsConstraints)


# --------------------------------------------------------------------------- #
# HeatConductionDAE (1-D transient heat conduction; DAE-HardNet Example 6).
# --------------------------------------------------------------------------- #
def test_heat_spec_counts() -> None:
    dae = HeatConductionDAE(alpha=0.5)
    s = dae.spec
    assert (s.n_differential, s.n_equality, s.n_inequality, s.n_outputs) == (1, 0, 2, 1)


def test_heat_differential_residual_is_heat_equation() -> None:
    """D: dT/dt - alpha * d2T/dx2, read from the derivative variables d."""
    alpha = 0.7
    dae = HeatConductionDAE(alpha=alpha)
    batch = 5
    x = torch.randn(batch, 1)
    t = torch.randn(batch, 1)
    y = torch.randn(batch, 1)  # T
    # d = [dT_dx, dT_dt, d2T_dx2, d2T_dt2]
    d = torch.randn(batch, 4)
    res = dae.differential(x, t, y, d)
    assert res.shape == (batch, 1)
    expected = d[:, 1:2] - alpha * d[:, 2:3]
    assert torch.allclose(res, expected)


def test_heat_inequality_temperature_bounds() -> None:
    """g = [T_min - T, T - T_max] <= 0 inside the band."""
    dae = HeatConductionDAE(alpha=1.0, T_min=15.0, T_max=35.0)
    T = torch.tensor([[10.0], [25.0], [40.0]])  # below / inside / above
    g = dae.inequality(torch.zeros(3, 1), torch.zeros(3, 1), T)
    assert g.shape == (3, 2)
    # Inside the band both entries are <= 0; outside, one entry is > 0.
    assert (g[1] <= 0).all()
    assert g[0, 0] > 0  # T_min - 10 = 5 > 0 (lower violation)
    assert g[2, 1] > 0  # 40 - T_max = 5 > 0 (upper violation)


def test_heat_violation_zero_when_satisfied() -> None:
    """The aggregate violation is ~0 when D=0 and T is within bounds."""
    alpha = 1.0
    dae = HeatConductionDAE(alpha=alpha, T_min=0.0, T_max=100.0)
    batch = 4
    x = torch.zeros(batch, 1)
    t = torch.zeros(batch, 1)
    y = torch.full((batch, 1), 25.0)  # within bounds
    # Choose d so that dT_dt == alpha * d2T_dx2 (heat eq satisfied).
    d = torch.zeros(batch, 4)
    d[:, 1] = 2.0  # dT_dt
    d[:, 2] = 2.0 / alpha  # d2T_dx2
    viol = dae.violation(x, t, y, d)
    assert viol.shape == ()
    assert viol.item() < 1e-6


# --------------------------------------------------------------------------- #
# MechanicalDAE (Euler-Lagrange; unconstrained n_holonomic=0).
# --------------------------------------------------------------------------- #
def _diag_inertia(n: int):
    def fn(q: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(n, device=q.device, dtype=q.dtype)
        return eye.unsqueeze(0).expand(q.shape[0], n, n)

    return fn


def _zero_force(n: int):
    def fn(*args: torch.Tensor) -> torch.Tensor:
        q = args[0]
        return torch.zeros(q.shape[0], n, device=q.device, dtype=q.dtype)

    return fn


def _unit_jacobian(n: int, m: int):
    """Constant holonomic Jacobian J(q) -> [batch, m, n]."""
    base = torch.zeros(m, n)
    for i in range(m):
        base[i, i] = 1.0
        if i + 1 < n:
            base[i, i + 1] = -1.0

    def fn(q: torch.Tensor) -> torch.Tensor:
        return base.to(q.dtype).unsqueeze(0).expand(q.shape[0], m, n)

    return fn


def _mechanical(n: int, m: int, q_bounds=None) -> MechanicalDAE:
    return MechanicalDAE(
        n_joints=n,
        n_holonomic=m,
        inertia_fn=_diag_inertia(n),
        coriolis_fn=_zero_force(n),
        gravity_fn=_zero_force(n),
        actuator_fn=lambda q: torch.zeros(q.shape[0], n, max(m, 1)),
        constraint_fn=_unit_jacobian(n, m) if m > 0 else None,
        q_bounds=q_bounds,
    )


def test_mechanical_spec_unconstrained() -> None:
    n = 2
    dae = _mechanical(n, 0)
    s = dae.spec
    # No holonomic constraints -> the EOM differential is n_joints-wide.
    assert s.n_differential == n
    assert s.n_equality == 0
    assert s.n_outputs == 2 * n  # y = [q, q_dot]


def test_mechanical_spec_matches_residual_widths() -> None:
    """ConstraintSpec counts MUST equal the actual residual vector widths.

    The KKT projection sizes its system from the spec, so any mismatch (which
    previously existed: the EOM residual is n_joints-wide but n_differential was
    1) would malform the projection on mechanical systems.
    """
    n, m = 2, 1
    b = 4
    q = torch.randn(b, n)
    q_dot = torch.randn(b, n)
    y = torch.cat([q, q_dot], dim=-1)
    t = torch.zeros(b, 1)
    q_bounds = (torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]))
    for nm in (0, m):
        dae = _mechanical(n, nm, q_bounds=q_bounds)
        s = dae.spec
        n_lam = nm
        d = (
            torch.cat([q_dot, torch.randn(b, n), torch.randn(b, n_lam)], dim=-1)
            if nm > 0
            else torch.cat([q_dot, torch.randn(b, n)], dim=-1)
        )
        assert dae.differential(q, t, y, d).shape[-1] == s.n_differential
        assert dae.equality(q, t, y).shape[-1] == s.n_equality
        assert dae.inequality(q, t, y).shape[-1] == s.n_inequality


def test_mechanical_holonomic_differential_blocks() -> None:
    """With m>0 the differential residual is [EOM(n), Psi(m), J q_dot(m)]."""
    n, m = 2, 1
    dae = _mechanical(n, m)
    b = 5
    q = torch.randn(b, n)
    q_dot = torch.randn(b, n)
    y = torch.cat([q, q_dot], dim=-1)
    q_ddot = torch.randn(b, n)
    lam = torch.randn(b, m)
    d = torch.cat([q_dot, q_ddot, lam], dim=-1)
    res = dae.differential(q, torch.zeros(b, 1), y, d)
    assert res.shape == (b, n + 2 * m)
    # The holonomic position block Psi = J q and velocity block J q_dot.
    J = _unit_jacobian(n, m)(q)
    psi = (J @ q.unsqueeze(-1)).squeeze(-1)
    jqdot = (J @ q_dot.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(res[:, n : n + m], psi, atol=1e-6)
    assert torch.allclose(res[:, n + m : n + 2 * m], jqdot, atol=1e-6)


def test_mechanical_eom_residual_free_motion() -> None:
    """For M=I, C=G=0 and no holonomic constraint, the EOM residual is just q_ddot."""
    n = 2
    dae = MechanicalDAE(
        n_joints=n,
        n_holonomic=0,
        inertia_fn=_diag_inertia(n),
        coriolis_fn=_zero_force(n),
        gravity_fn=_zero_force(n),
        actuator_fn=lambda q: torch.zeros(q.shape[0], n, n),
    )
    batch = 6
    q = torch.randn(batch, n)
    q_dot = torch.randn(batch, n)
    y = torch.cat([q, q_dot], dim=-1)  # [q, q_dot]
    q_ddot = torch.randn(batch, n)
    d = torch.cat([q_dot, q_ddot], dim=-1)  # [q_dot, q_ddot]
    res = dae.differential(q, torch.zeros(batch, 1), y, d)
    # M q_ddot + C + G - B u - J^T lambda = I q_ddot + 0 = q_ddot.
    assert res.shape == (batch, n)
    assert torch.allclose(res, q_ddot, atol=1e-6)


def test_mechanical_joint_limits_inequality() -> None:
    n = 2
    q_min = torch.tensor([-1.0, -1.0])
    q_max = torch.tensor([1.0, 1.0])
    dae = MechanicalDAE(
        n_joints=n,
        n_holonomic=0,
        inertia_fn=_diag_inertia(n),
        coriolis_fn=_zero_force(n),
        gravity_fn=_zero_force(n),
        actuator_fn=lambda q: torch.zeros(q.shape[0], n, n),
        q_bounds=(q_min, q_max),
    )
    assert dae.spec.n_inequality == 2 * n
    q = torch.tensor([[2.0, 0.0]])  # joint 0 over its upper limit
    q_dot = torch.zeros(1, n)
    y = torch.cat([q, q_dot], dim=-1)
    g = dae.inequality(q, torch.zeros(1, 1), y)
    assert g.shape == (1, 2 * n)
    assert (g > 0).any()  # the over-limit joint violates g <= 0
