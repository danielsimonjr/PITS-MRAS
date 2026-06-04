"""Nonlinear plant models for the PITS-MRAS examples.

Each example's closed loop controls a *plant*. These replace the earlier toy
linear surrogate steps with physically-faithful **nonlinear** dynamics, so the
demos exercise the controller's model-mismatch robustness (the point of adaptive
control). Each model is designed so its small-signal linearization recovers the
example's original linear surrogate -- the LQR/CBF controller (designed on the
linear reference model) therefore stays stabilizing near the operating point,
while the nonlinearity adds fidelity away from it.

All steps are pure functions: ``f(state, u, dt, **params) -> next_state`` with
``state`` / return a 1-D float32 ``Tensor`` of shape ``[2]`` and ``u`` a scalar.
Integration is semi-implicit (symplectic) Euler for the mechanical models.
"""

from __future__ import annotations

import torch
from torch import Tensor


def pendulum_step(
    state: Tensor,
    u: float,
    dt: float,
    g_over_l: float = 4.0,
    damping: float = 4.0,
) -> Tensor:
    r"""Nonlinear 1-DOF manipulator joint (rigid pendulum, ``m l^2 = 1``).

    :math:`\ddot\theta = u - (g/l)\sin\theta - b\dot\theta`. The ``\sin`` gravity
    term is the nonlinearity; :math:`\theta = 0` (hanging) is the stable
    equilibrium. Linearizes (``sin θ ≈ θ``) to ``u - g_over_l·θ - damping·θ̇``,
    recovering the manipulator example's linear surrogate at the defaults.

    Args:
        state: ``[theta, theta_dot]``.
        u: applied joint torque.
        dt: time step.
        g_over_l: gravity / length ratio (linear restoring stiffness).
        damping: viscous joint damping.
    """
    theta = float(state[0])
    theta_dot = float(state[1])
    theta_ddot = u - g_over_l * torch.sin(torch.tensor(theta)).item() - damping * theta_dot
    theta_dot_n = theta_dot + dt * theta_ddot
    theta_n = theta + dt * theta_dot_n
    return torch.tensor([theta_n, theta_dot_n], dtype=torch.float32)


def lateral_tyre_step(
    state: Tensor,
    u: float,
    dt: float,
    tyre_stiffness: float = 2.0,
    damping: float = 3.0,
    gust: float = 0.0,
) -> Tensor:
    r"""Single-track lateral dynamics with tyre-force saturation.

    :math:`\ddot y = u - k\tanh(y) - c\dot y + w`, where ``k tanh(y)`` is the
    lateral restoring force that **saturates** at :math:`\pm k` (a real tyre
    cannot produce unbounded lateral force) and ``w`` is a wind gust acting on
    the plant. Linearizes (``tanh y ≈ y``) to ``u - k·y - c·ẏ + gust``,
    recovering the AV example's linear surrogate at the defaults.

    Args:
        state: ``[y, y_dot]`` (lateral offset, lateral velocity).
        u: lateral control.
        dt: time step.
        tyre_stiffness: small-signal lateral stiffness ``k`` (saturation level).
        damping: lateral damping ``c``.
        gust: external lateral disturbance acting on the plant.
    """
    y = float(state[0])
    y_dot = float(state[1])
    restoring = tyre_stiffness * torch.tanh(torch.tensor(y)).item()
    y_ddot = u - restoring - damping * y_dot + gust
    y_dot_n = y_dot + dt * y_ddot
    y_n = y + dt * y_dot_n
    return torch.tensor([y_n, y_dot_n], dtype=torch.float32)


def rc_thermal_step(
    state: Tensor,
    u: float,
    dt: float,
    a_zone_mass: float = 2.0,
    a_zone_ambient: float = 1.0,
    a_mass_zone: float = 1.0,
    heater_gain: float = 3.0,
    u_max: float = 5.0,
) -> Tensor:
    r"""Two-node RC building-thermal network with a saturated heater.

    Deviation-from-ambient coordinates ``[T_z, T_m]`` (zone, thermal-mass
    temperatures), so the ambient is ``0``:

    .. math::
        \dot T_z = a_{zm}(T_m - T_z) - a_{za} T_z + b\,\mathrm{sat}(u),\quad
        \dot T_m = a_{mz}(T_z - T_m).

    The RC network is linear and Hurwitz; the **only** nonlinearity is heater
    saturation ``sat(u) = clip(u, -u_max, u_max)``. The HVAC example's reference
    model is the RC linearization, so the LQR tracks the zone temperature.

    Args:
        state: ``[T_zone, T_mass]`` deviations from ambient.
        u: commanded heating/cooling power (saturated to ``[-u_max, u_max]``).
        dt: time step.
        a_zone_mass, a_zone_ambient, a_mass_zone: RC conductance/capacitance
            coefficients.
        heater_gain: actuator gain ``b``.
        u_max: heater saturation magnitude.
    """
    t_zone = float(state[0])
    t_mass = float(state[1])
    u_sat = max(-u_max, min(u_max, u))
    t_zone_dot = (
        a_zone_mass * (t_mass - t_zone)
        - a_zone_ambient * t_zone
        + heater_gain * u_sat
    )
    t_mass_dot = a_mass_zone * (t_zone - t_mass)
    t_zone_n = t_zone + dt * t_zone_dot
    t_mass_n = t_mass + dt * t_mass_dot
    return torch.tensor([t_zone_n, t_mass_n], dtype=torch.float32)
