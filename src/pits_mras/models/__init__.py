"""Models subpackage: attention, port-Hamiltonian decoders, critic/costate, PITNN.

Owning phase: Phase 2 (Neural Network Models) per ROADMAP.md / ARCHITECTURE.md
§2.1. Now that the Phase-2 backbone is implemented, the six public model
classes are re-exported for convenient import.
"""

from pits_mras.models.adversary import NeuralAdversary
from pits_mras.models.attention import PhysicsInformedAttention
from pits_mras.models.critic import AdversaryHead, CostateHead, QuadraticCritic
from pits_mras.models.decoders import (
    DissipationNet,
    HamiltonianNet,
    PortHamiltonianDecoder,
)
from pits_mras.models.generic import GFINNDecoder
from pits_mras.models.koopman import KoopmanLiftingModel, koopman_loss
from pits_mras.models.pitnn import PITNN
from pits_mras.models.sac import GaussianPolicy, TwinQCritic
from pits_mras.models.tdmpc import LatentModel, MPPIPlanner, WorldModel

__all__ = [
    "PhysicsInformedAttention",
    "HamiltonianNet",
    "DissipationNet",
    "PortHamiltonianDecoder",
    "GFINNDecoder",
    "QuadraticCritic",
    "CostateHead",
    "AdversaryHead",
    "NeuralAdversary",
    "PITNN",
    "KoopmanLiftingModel",
    "koopman_loss",
    "GaussianPolicy",
    "TwinQCritic",
    "WorldModel",
    "MPPIPlanner",
    "LatentModel",
]
