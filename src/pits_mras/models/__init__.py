"""Models subpackage: attention, port-Hamiltonian decoders, critic/costate, PITNN.

Owning phase: Phase 2 (Neural Network Models) per ROADMAP.md / ARCHITECTURE.md
§2.1. Now that the Phase-2 backbone is implemented, the six public model
classes are re-exported for convenient import.
"""

from pits_mras.models.attention import PhysicsInformedAttention
from pits_mras.models.critic import CostateHead, QuadraticCritic
from pits_mras.models.decoders import (
    DissipationNet,
    HamiltonianNet,
    PortHamiltonianDecoder,
)
from pits_mras.models.pitnn import PITNN

__all__ = [
    "PhysicsInformedAttention",
    "HamiltonianNet",
    "DissipationNet",
    "PortHamiltonianDecoder",
    "QuadraticCritic",
    "CostateHead",
    "PITNN",
]
