"""Physics constraint systems for PCML (PCML Addendum §2.1).

Plug-in DAE constraint definitions consumed by the soft PCML loss and the hard
KKT projection layer. Each system exposes differential / equality / inequality
residuals via the :class:`PhysicsConstraints` interface.
"""

from pits_mras.constraints.base import ConstraintSpec, PhysicsConstraints
from pits_mras.constraints.mechanical import MechanicalDAE
from pits_mras.constraints.thermal import HeatConductionDAE

__all__ = [
    "ConstraintSpec",
    "PhysicsConstraints",
    "MechanicalDAE",
    "HeatConductionDAE",
]
