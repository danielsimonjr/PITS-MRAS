"""Thermal-system constraints for HVAC / heat conduction (PCML Addendum §2.1).

1-D transient heat conduction (DAE-HardNet Example 6)::

    dT/dt = alpha * d2T/dx2          (heat diffusion)
    T_min <= T <= T_max              (operational bounds)

For the PITS-MRAS HVAC application ``x`` is the spatial coordinate, ``t`` is
time, ``y = T(x, t)`` is the temperature field, and ``alpha`` is the thermal
diffusivity. The differential residual reads the derivative variables ``d`` so
the PDE constraint becomes algebraic in the KKT projection.
"""

import torch
from torch import Tensor

from pits_mras.constraints.base import ConstraintSpec, PhysicsConstraints


class HeatConductionDAE(PhysicsConstraints):
    """1-D transient heat conduction ``dT/dt = alpha * d2T/dx2``.

    Derivative-variable layout ``d = [dT_dx, dT_dt, d2T_dx2, d2T_dt2]``.
    """

    def __init__(self, alpha: float, T_min: float = 15.0, T_max: float = 35.0) -> None:
        self.alpha = alpha
        self.T_min = T_min
        self.T_max = T_max
        self._spec = ConstraintSpec(
            n_differential=1,  # heat-equation residual
            n_equality=0,
            n_inequality=2,  # T_min <= T <= T_max
            n_outputs=1,
        )

    @property
    def spec(self) -> ConstraintSpec:
        return self._spec

    def differential(self, x: Tensor, t: Tensor, y: Tensor, d: Tensor) -> Tensor:
        """``D: dT/dt - alpha * d2T/dx2`` -> ``[batch, 1]``."""
        dT_dt = d[:, 1:2]  # time derivative
        d2T_dx2 = d[:, 2:3]  # second spatial derivative
        return dT_dt - self.alpha * d2T_dx2

    def equality(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """No algebraic equality constraints -> ``[batch, 0]``."""
        return torch.zeros(x.shape[0], 0, device=x.device, dtype=x.dtype)

    def inequality(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """``g = [T_min - T, T - T_max]`` -> ``[batch, 2]`` (``<= 0`` feasible)."""
        T = y[:, 0:1]
        lower = self.T_min - T  # <= 0 when T >= T_min
        upper = T - self.T_max  # <= 0 when T <= T_max
        return torch.cat([lower, upper], dim=-1)
