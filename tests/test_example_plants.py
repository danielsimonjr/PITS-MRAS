"""Physics-sanity tests for the nonlinear example plants (`examples/plants.py`).

Each plant's small-signal linearization recovers the example's original linear
surrogate; here we check the nonlinear behavior (gravity sin, tyre tanh
saturation, RC relaxation + heater saturation) and finiteness.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

import torch

_PLANTS_PATH = pathlib.Path(__file__).resolve().parent.parent / "examples" / "plants.py"


def _load_plants():
    spec = importlib.util.spec_from_file_location("_pits_example_plants", _PLANTS_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


plants = _load_plants()


# --------------------------------------------------------------------------- #
# Pendulum (nonlinear manipulator joint).
# --------------------------------------------------------------------------- #
def test_pendulum_rest_stays_at_rest() -> None:
    """(theta, theta_dot) = 0 with u = 0 is the equilibrium -> no motion."""
    nxt = plants.pendulum_step(torch.zeros(2), 0.0, dt=0.01)
    assert torch.allclose(nxt, torch.zeros(2), atol=1e-7)


def test_pendulum_damping_decays_velocity() -> None:
    """With no input, a moving joint loses speed (viscous damping)."""
    state = torch.tensor([0.0, 1.0])
    nxt = plants.pendulum_step(state, 0.0, dt=0.01, g_over_l=4.0, damping=4.0)
    assert abs(float(nxt[1])) < 1.0  # |theta_dot| decreased
    assert torch.isfinite(nxt).all()


def test_pendulum_gravity_saturates_vs_linear() -> None:
    """The sin gravity restoring is weaker than the linear term for large angle:
    at theta=1.0, (g/l) sin(1) < (g/l)*1."""
    g_over_l = 4.0
    theta = 1.0
    # One step from rest at theta with u=0, dt tiny -> theta_ddot ~ -(g/l) sin th.
    nxt = plants.pendulum_step(
        torch.tensor([theta, 0.0]), 0.0, dt=1e-3, g_over_l=g_over_l, damping=0.0
    )
    theta_ddot = (float(nxt[1]) - 0.0) / 1e-3
    assert math.isclose(theta_ddot, -g_over_l * math.sin(theta), rel_tol=1e-4)
    assert abs(theta_ddot) < g_over_l * theta  # saturating (< linear restoring)


# --------------------------------------------------------------------------- #
# Lateral tyre-saturation (AV).
# --------------------------------------------------------------------------- #
def test_lateral_rest_stays_at_rest() -> None:
    nxt = plants.lateral_tyre_step(torch.zeros(2), 0.0, dt=0.01, gust=0.0)
    assert torch.allclose(nxt, torch.zeros(2), atol=1e-7)


def test_lateral_tyre_force_saturates() -> None:
    """The lateral restoring magnitude is bounded by tyre_stiffness even for a
    large offset (tanh saturation), unlike an unbounded linear spring."""
    k = 2.0
    big_y = 100.0
    nxt = plants.lateral_tyre_step(
        torch.tensor([big_y, 0.0]), 0.0, dt=1e-3, tyre_stiffness=k, damping=0.0
    )
    y_ddot = (float(nxt[1]) - 0.0) / 1e-3
    assert abs(y_ddot) <= k + 1e-5  # |restoring| <= k (saturated)


def test_lateral_gust_perturbs_velocity() -> None:
    """A gust changes the lateral velocity in its direction."""
    nxt = plants.lateral_tyre_step(torch.zeros(2), 0.0, dt=0.01, gust=1.0)
    assert float(nxt[1]) > 0.0
    assert torch.isfinite(nxt).all()


# --------------------------------------------------------------------------- #
# RC thermal network (HVAC).
# --------------------------------------------------------------------------- #
def test_rc_heater_saturates() -> None:
    """Commanding beyond u_max gives the same zone response as u_max."""
    s = torch.zeros(2)
    a = plants.rc_thermal_step(s, 100.0, dt=0.01, u_max=5.0)
    b = plants.rc_thermal_step(s, 5.0, dt=0.01, u_max=5.0)
    assert torch.allclose(a, b, atol=1e-7)


def test_rc_relaxes_toward_ambient() -> None:
    """With no heating, a warm zone cools toward ambient (0 in deviation)."""
    s = torch.tensor([2.0, 2.0])
    nxt = plants.rc_thermal_step(s, 0.0, dt=0.01)
    assert float(nxt[0]) < 2.0  # zone temperature dropped
    assert torch.isfinite(nxt).all()


def test_rc_zone_mass_coupling() -> None:
    """A hot zone warms the cooler thermal mass (coupling moves T_m up)."""
    s = torch.tensor([5.0, 0.0])
    nxt = plants.rc_thermal_step(s, 0.0, dt=0.01)
    assert float(nxt[1]) > 0.0  # mass warmed by the zone
