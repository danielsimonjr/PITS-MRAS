"""Mechanical-system constraints: Euler-Lagrange DAEs (PCML Addendum §2.1).

For a robot with ``n`` joints and ``m`` holonomic constraints::

    D1: M(q) q_ddot + C(q, q_dot) + G(q) - J(q)^T lambda = 0   (equations of motion)
    D2: Psi(q) = J(q) q = 0                                    (holonomic position)
    D3: J(q) q_dot = 0                                         (velocity constraint)
    g:  q_min <= q <= q_max                                    (joint limits)
    g:  u_min <= u <= u_max                                    (torque limits)

The state is ``y = [q, q_dot]`` and the derivative variables are
``d = [q_dot, q_ddot]`` (unconstrained) or ``d = [q_dot, q_ddot, lambda]``
(holonomic). The holonomic position/velocity constraints are folded into the
differential block (``D2``/``D3``) and the constraint force enters ``D1`` via
``J^T lambda``; there is therefore no separate algebraic equality residual
(``n_equality = 0``), keeping the spec self-consistent for the KKT projection.

The control ``u`` is not part of the constraint signature -- the residual
constrains the predicted *dynamics* ``[q_dot, q_ddot, lambda]`` to be consistent
with the equations of motion; any applied torque is supplied externally.
"""

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from pits_mras.constraints.base import ConstraintSpec, PhysicsConstraints


class MechanicalDAE(PhysicsConstraints):
    """Euler-Lagrange DAE constraints with optional holonomic constraints.

    Args:
        n_joints: number of generalized coordinates ``n``.
        n_holonomic: number of holonomic constraints ``m`` (0 = unconstrained).
        inertia_fn: ``q -> [batch, n, n]`` mass/inertia matrix ``M(q)``.
        coriolis_fn: ``(q, q_dot) -> [batch, n]`` Coriolis/centrifugal forces.
        gravity_fn: ``q -> [batch, n]`` gravity forces ``G(q)``.
        actuator_fn: ``q -> [batch, n, m_ctrl]`` actuator matrix ``B(q)``
            (kept for API completeness; not used in the residual).
        constraint_fn: ``q -> [batch, m, n]`` holonomic Jacobian ``J(q)``.
        q_bounds: optional ``(q_min, q_max)`` joint limits, each ``[n]``.
        u_bounds: optional ``(u_min, u_max)`` torque limits (counted in the spec).
    """

    def __init__(
        self,
        n_joints: int,
        n_holonomic: int,
        inertia_fn: Callable[[Tensor], Tensor],
        coriolis_fn: Callable[[Tensor, Tensor], Tensor],
        gravity_fn: Callable[[Tensor], Tensor],
        actuator_fn: Callable[[Tensor], Tensor],
        constraint_fn: Optional[Callable[[Tensor], Tensor]] = None,
        q_bounds: Optional[Tuple[Tensor, Tensor]] = None,
        u_bounds: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> None:
        self.n_joints = n_joints
        self.n_holonomic = n_holonomic
        self.inertia_fn = inertia_fn
        self.coriolis_fn = coriolis_fn
        self.gravity_fn = gravity_fn
        self.actuator_fn = actuator_fn
        self.constraint_fn = constraint_fn
        self.q_bounds = q_bounds
        self.u_bounds = u_bounds

        constrained = n_holonomic > 0 and constraint_fn is not None
        n_diff = 1 + (2 if constrained else 0)  # EOM (+ Psi=0, J q_dot=0)
        n_ineq = (2 * n_joints if q_bounds is not None else 0) + (
            2 * n_joints if u_bounds is not None else 0
        )
        self._spec = ConstraintSpec(
            n_differential=n_diff,
            n_equality=0,
            n_inequality=n_ineq,
            n_outputs=n_joints,
        )

    @property
    def spec(self) -> ConstraintSpec:
        return self._spec

    def differential(self, x: Tensor, t: Tensor, y: Tensor, d: Tensor) -> Tensor:
        """EOM (+ holonomic) residual -> ``[batch, n_differential]``.

        ``y = [q, q_dot]``; ``d = [q_dot, q_ddot(, lambda)]``.
        """
        n = self.n_joints
        q = y[:, :n]
        q_dot = y[:, n : 2 * n]
        q_ddot = d[:, n : 2 * n]

        M = self.inertia_fn(q)  # [batch, n, n]
        C = self.coriolis_fn(q, q_dot)  # [batch, n]
        G = self.gravity_fn(q)  # [batch, n]
        eom = (M @ q_ddot.unsqueeze(-1)).squeeze(-1) + C + G  # [batch, n]

        constrained = self.n_holonomic > 0 and self.constraint_fn is not None
        if constrained:
            assert self.constraint_fn is not None  # for type-checkers
            J = self.constraint_fn(q)  # [batch, m, n]
            lam = d[:, 2 * n : 2 * n + self.n_holonomic]  # [batch, m]
            eom = eom - (J.transpose(-1, -2) @ lam.unsqueeze(-1)).squeeze(-1)
            psi = (J @ q.unsqueeze(-1)).squeeze(-1)  # Psi(q) = J q  [batch, m]
            j_qdot = (J @ q_dot.unsqueeze(-1)).squeeze(-1)  # J q_dot  [batch, m]
            return torch.cat([eom, psi, j_qdot], dim=-1)
        return eom

    def equality(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """No separate algebraic equality residual -> ``[batch, 0]``.

        (Holonomic position/velocity constraints live in the differential block;
        the constraint force enters there via ``J^T lambda``.)
        """
        return torch.zeros(y.shape[0], 0, device=y.device, dtype=y.dtype)

    def inequality(self, x: Tensor, t: Tensor, y: Tensor) -> Tensor:
        """Joint-limit residual ``g = [q_min - q, q - q_max] <= 0`` -> ``[batch, 2n]``.

        (Torque limits are counted in the spec but enforced on the applied
        control externally, since ``u`` is not part of this signature.)
        """
        n = self.n_joints
        q = y[:, :n]
        if self.q_bounds is None:
            return torch.zeros(y.shape[0], 0, device=y.device, dtype=y.dtype)
        q_min, q_max = self.q_bounds
        q_min = q_min.to(device=q.device, dtype=q.dtype)
        q_max = q_max.to(device=q.device, dtype=q.dtype)
        return torch.cat([q_min - q, q - q_max], dim=-1)
