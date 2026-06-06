"""PITS-MRAS: Physics-Informed Time-Series Model-Reference Adaptive Systems.

A unified framework merging Physics-Informed Neural Networks (PINNs),
Time-Series Deep Learning, and Model-Reference Adaptive Control (MRAS).

Mathematical foundation: The MRAS Lyapunov function V(e)=eᵀPe is the LQR
value function for the tracking-error system; policy iteration on the CARE
(Kleinman 1968) is the formal backbone; IRL (Vrabie & Lewis 2009) makes it
model-free. The port-Hamiltonian decoder makes H_θ a storage/value function
(passivity = L2-gain), and the costate λ=∇V is enforced architecturally.

Top-level symbol re-exports (IP §4.1): the design plan lists eight public
symbols. The six class symbols are re-exported here, and (since Phase 5 landed)
so are the two *function* symbols ``pretrain_pitnn`` (training/pretrain.py) and
``cotraining_loop`` (training/cotrain.py).
"""

from pits_mras.constraints import (
    ConstraintSpec,
    HeatConductionDAE,
    MechanicalDAE,
    PhysicsConstraints,
)
from pits_mras.controllers.mras import MRASController
from pits_mras.controllers.reference_models import LinearReferenceModel
from pits_mras.controllers.safety import CLFCBFSafetyFilter
from pits_mras.inference.realtime import RealtimeInferenceEngine
from pits_mras.models.critic import QuadraticCritic
from pits_mras.models.lagrangian_head import LagrangianMultiplierHead
from pits_mras.models.pcml import (
    KKTProjectionLayer,
    PCMLModule,
    SoftPCMLLoss,
    TaylorNeighborhoodApproximation,
)
from pits_mras.models.pitnn import PITNN

# Phase 5 (DONE): the training pipelines are implemented, so the two function
# symbols are now re-exported on the public package surface alongside the six
# class symbols above. (Deferred since Phase 1 to keep ``import pits_mras``
# working while the training modules were docstring-only stubs.)
from pits_mras.training import cotraining_loop, pretrain_pitnn

__version__ = "0.5.3"

__all__ = [
    "PITNN",
    "QuadraticCritic",
    "MRASController",
    "LinearReferenceModel",
    "CLFCBFSafetyFilter",
    "RealtimeInferenceEngine",
    "pretrain_pitnn",
    "cotraining_loop",
    # PCML (Physics-Constrained ML) -- constraints + soft/hard enforcement.
    "PhysicsConstraints",
    "ConstraintSpec",
    "MechanicalDAE",
    "HeatConductionDAE",
    "SoftPCMLLoss",
    "TaylorNeighborhoodApproximation",
    "KKTProjectionLayer",
    "PCMLModule",
    "LagrangianMultiplierHead",
]
