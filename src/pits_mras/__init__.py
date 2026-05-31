"""PITS-MRAS: Physics-Informed Time-Series Model-Reference Adaptive Systems.

A unified framework merging Physics-Informed Neural Networks (PINNs),
Time-Series Deep Learning, and Model-Reference Adaptive Control (MRAS).

Mathematical foundation: The MRAS Lyapunov function V(e)=eᵀPe is the LQR
value function for the tracking-error system; policy iteration on the CARE
(Kleinman 1968) is the formal backbone; IRL (Vrabie & Lewis 2009) makes it
model-free. The port-Hamiltonian decoder makes H_θ a storage/value function
(passivity = L2-gain), and the costate λ=∇V is enforced architecturally.

Top-level symbol re-exports (IP §4.1): the design plan lists eight public
symbols. As of Phase 1, the six class symbols are importable (their modules
ship stub classes), so they are re-exported here. The two *function* symbols
``pretrain_pitnn`` (training/pretrain.py) and ``cotraining_loop``
(training/cotrain.py) do NOT yet exist as importable names — those modules are
docstring-only stubs — so re-exporting them now would break ``import
pits_mras``. They are intentionally deferred (see TODO below) and will be added
once Phase 5 lands.
"""

from pits_mras.controllers.mras import MRASController
from pits_mras.controllers.reference_models import LinearReferenceModel
from pits_mras.controllers.safety import CLFCBFSafetyFilter
from pits_mras.inference.realtime import RealtimeInferenceEngine
from pits_mras.models.critic import QuadraticCritic
from pits_mras.models.pitnn import PITNN

# TODO(phase-5): re-export pretrain_pitnn (training/pretrain.py) and
# cotraining_loop (training/cotrain.py) once those functions are implemented.
# They are absent from the catalog below to keep ``import pits_mras`` working.

__version__ = "0.1.0"

__all__ = [
    "PITNN",
    "QuadraticCritic",
    "MRASController",
    "LinearReferenceModel",
    "CLFCBFSafetyFilter",
    "RealtimeInferenceEngine",
]
