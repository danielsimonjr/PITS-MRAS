# pits_mras - Dependency Graph

Comprehensive dependency graph of all Python modules, imports, exports, functions, classes, and constants in the codebase.

---

## Overview

The codebase is organized into the following modules:

- **examples**: 5 files
- **root**: 1 file
- **src/pits_mras**: 2 files
- **src/pits_mras/constraints**: 4 files
- **src/pits_mras/controllers**: 5 files
- **src/pits_mras/data**: 2 files
- **src/pits_mras/inference**: 3 files
- **src/pits_mras/losses**: 7 files
- **src/pits_mras/models**: 12 files
- **src/pits_mras/training**: 7 files
- **src/pits_mras/utils**: 7 files

---

## Examples Dependencies

### `examples/autonomous_vehicle.py` - Example: autonomous vehicle lateral control (IP §10.2).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `numpy` | `(module)` |
| `torch` | `(module)` |
| `torch` | `(module)` |
| `plants` | `lateral_tyre_step` |
| `matplotlib` | `(module)` |
| `matplotlib.pyplot` | `(module)` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |
| `math` | `(module)` |
| `typing` | `Any` |
| `pathlib` | `(module)` |
| `sys` | `(module)` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/config.py` | `NetworkConfig, PhysicsConfig, PITSMRASConfig` | Import |
| `src/pits_mras/controllers/mras.py` | `MRASController` | Import |
| `src/pits_mras/controllers/reference_models.py` | `LinearReferenceModel` | Import |
| `src/pits_mras/inference/realtime.py` | `RealtimeInferenceEngine` | Import |
| `src/pits_mras/models/__init__.py` | `PITNN` | Import |

**Exports:**
- Functions: `run`, `main`

---

### `examples/building_hvac.py` - Example: building HVAC thermal-zone control (IP §10.3).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `plants` | `rc_thermal_step` |
| `numpy` | `(module)` |
| `torch` | `(module)` |
| `matplotlib` | `(module)` |
| `matplotlib.pyplot` | `(module)` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |
| `math` | `(module)` |
| `typing` | `Any` |
| `pathlib` | `(module)` |
| `sys` | `(module)` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/config.py` | `NetworkConfig, PhysicsConfig, PITSMRASConfig` | Import |
| `src/pits_mras/controllers/mras.py` | `MRASController` | Import |
| `src/pits_mras/controllers/reference_models.py` | `LinearReferenceModel` | Import |
| `src/pits_mras/inference/realtime.py` | `RealtimeInferenceEngine` | Import |
| `src/pits_mras/models/__init__.py` | `PITNN` | Import |

**Exports:**
- Functions: `run`, `main`

---

### `examples/pcml_heat_diffusion.py` - Example: hard PCML on the 1-D heat equation with real coordinates.

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `matplotlib` | `(module)` |
| `torch` | `(module)` |
| `matplotlib.pyplot` | `(module)` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |
| `math` | `(module)` |
| `typing` | `Any` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/constraints/__init__.py` | `HeatConductionDAE` | Import |
| `src/pits_mras/models/pcml.py` | `PCMLModule, SoftPCMLLoss` | Import |

**Exports:**
- Functions: `run`, `main`

---

### `examples/plants.py` - Nonlinear plant models for the PITS-MRAS examples.

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |

**Exports:**
- Functions: `pendulum_step`, `lateral_tyre_step`, `rc_thermal_step`

---

### `examples/robotic_manipulator.py` - Example: 2-DOF planar robotic manipulator (IP §10.1).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `matplotlib` | `(module)` |
| `numpy` | `(module)` |
| `torch` | `(module)` |
| `plants` | `pendulum_step` |
| `matplotlib.pyplot` | `(module)` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |
| `math` | `(module)` |
| `typing` | `Any` |
| `pathlib` | `(module)` |
| `sys` | `(module)` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/config.py` | `NetworkConfig, PhysicsConfig, PITSMRASConfig` | Import |
| `src/pits_mras/controllers/mras.py` | `MRASController` | Import |
| `src/pits_mras/controllers/reference_models.py` | `LinearReferenceModel` | Import |
| `src/pits_mras/inference/realtime.py` | `RealtimeInferenceEngine` | Import |
| `src/pits_mras/models/__init__.py` | `PITNN` | Import |
| `src/pits_mras/training/irl_trainer.py` | `train_irl_critic_gd` | Import |

**Exports:**
- Functions: `run`, `main`

---

## Root Dependencies

### `setup.py` - setup module

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `setuptools` | `find_packages, setup` |

---

## Src / pits_mras Dependencies

### `src/pits_mras/__init__.py` - PITS-MRAS: Physics-Informed Time-Series Model-Reference Adaptive Systems.

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/constraints/__init__.py` | `ConstraintSpec, HeatConductionDAE, MechanicalDAE, PhysicsConstraints` | Re-export |
| `src/pits_mras/controllers/mras.py` | `MRASController` | Re-export |
| `src/pits_mras/controllers/reference_models.py` | `LinearReferenceModel` | Re-export |
| `src/pits_mras/controllers/safety.py` | `CLFCBFSafetyFilter` | Re-export |
| `src/pits_mras/inference/realtime.py` | `RealtimeInferenceEngine` | Re-export |
| `src/pits_mras/models/critic.py` | `QuadraticCritic` | Re-export |
| `src/pits_mras/models/lagrangian_head.py` | `LagrangianMultiplierHead` | Re-export |
| `src/pits_mras/models/pcml.py` | `KKTProjectionLayer, PCMLModule, SoftPCMLLoss, TaylorNeighborhoodApproximation` | Re-export |
| `src/pits_mras/models/pitnn.py` | `PITNN` | Re-export |
| `src/pits_mras/training/__init__.py` | `cotraining_loop, pretrain_pitnn` | Re-export |

**Exports:**
- Re-exports: `ConstraintSpec`, `HeatConductionDAE`, `MechanicalDAE`, `PhysicsConstraints`, `MRASController`, `LinearReferenceModel`, `CLFCBFSafetyFilter`, `RealtimeInferenceEngine`, `QuadraticCritic`, `LagrangianMultiplierHead`, `KKTProjectionLayer`, `PCMLModule`, `SoftPCMLLoss`, `TaylorNeighborhoodApproximation`, `PITNN`, `cotraining_loop`, `pretrain_pitnn`

---

### `src/pits_mras/config.py` - Centralized configuration for PITS-MRAS (IP §4.2).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `yaml` | `(module)` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `dataclasses` | `(module)` |
| `dataclasses` | `dataclass, field` |
| `typing` | `List, Optional` |

**Exports:**
- Classes: `NetworkConfig`, `PhysicsConfig`, `MRASConfig`, `SafetyConfig`, `LossConfig`, `TrainingConfig`, `PCMLConfig`, `PITSMRASConfig`

---

## Src / pits_mras / constraints Dependencies

### `src/pits_mras/constraints/__init__.py` - Physics constraint systems for PCML (PCML Addendum §2.1).

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/constraints/base.py` | `ConstraintSpec, PhysicsConstraints` | Re-export |
| `src/pits_mras/constraints/mechanical.py` | `MechanicalDAE` | Re-export |
| `src/pits_mras/constraints/thermal.py` | `HeatConductionDAE` | Re-export |

**Exports:**
- Re-exports: `ConstraintSpec`, `PhysicsConstraints`, `MechanicalDAE`, `HeatConductionDAE`

---

### `src/pits_mras/constraints/base.py` - Base classes for physics constraint specifications (PCML Addendum §2.1).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `abc` | `ABC, abstractmethod` |
| `dataclasses` | `dataclass` |

**Exports:**
- Classes: `ConstraintSpec`
- Protocols/ABCs: `PhysicsConstraints`

---

### `src/pits_mras/constraints/mechanical.py` - Mechanical-system constraints: Euler-Lagrange DAEs (PCML Addendum §2.1).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `typing` | `Callable, Optional, Tuple` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/constraints/base.py` | `ConstraintSpec, PhysicsConstraints` | Import |

**Exports:**
- Classes: `MechanicalDAE`

---

### `src/pits_mras/constraints/thermal.py` - Thermal-system constraints for HVAC / heat conduction (PCML Addendum §2.1).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch` | `Tensor` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/constraints/base.py` | `ConstraintSpec, PhysicsConstraints` | Import |

**Exports:**
- Classes: `HeatConductionDAE`

---

## Src / pits_mras / controllers Dependencies

### `src/pits_mras/controllers/__init__.py` - Controllers subpackage: reference models, CLF-CBF safety filter, MRAS actor.

---

### `src/pits_mras/controllers/koopman_control.py` - Koopman-LQR controller (ROADMAP integration #5). NEW capability.

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `numpy` | `(module)` |
| `torch` | `(module)` |
| `torch.nn` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |
| `typing` | `Optional, Union` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/models/koopman.py` | `KoopmanLiftingModel` | Import |
| `src/pits_mras/utils/lyapunov.py` | `solve_care` | Import |

**Exports:**
- Classes: `KoopmanLQRController`

---

### `src/pits_mras/controllers/mras.py` - Actor-critic MRAS controller (IP §7.3). Identities 1, 2, 3, 4.

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `typing` | `Dict, Optional, Tuple` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/controllers/reference_models.py` | `LinearReferenceModel` | Import |
| `src/pits_mras/controllers/safety.py` | `CLFCBFSafetyFilter` | Import |
| `src/pits_mras/models/critic.py` | `CostateHead, QuadraticCritic` | Import |
| `src/pits_mras/utils/lyapunov.py` | `solve_care` | Import |

**Exports:**
- Classes: `MRASController`

---

### `src/pits_mras/controllers/reference_models.py` - Linear reference model (IP §7.1).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `numpy` | `(module)` |
| `torch` | `(module)` |
| `torch.nn` | `(module)` |
| `torch` | `Tensor` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/utils/lyapunov.py` | `check_hurwitz, kleinman_iteration, solve_lyapunov` | Import |

**Exports:**
- Classes: `LinearReferenceModel`

---

### `src/pits_mras/controllers/safety.py` - CLF-CBF-QP safety filter (IP §7.2 / §3.4). NEW -- Identity 3.

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch.nn` | `(module)` |
| `torch.nn.functional` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `typing` | `Tuple` |

**Exports:**
- Classes: `CLFCBFSafetyFilter`

---

## Src / pits_mras / data Dependencies

### `src/pits_mras/data/__init__.py` - Data subpackage: reusable trajectory dataset, generator, and loader (Gap G7).

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/data/trajectory.py` | `TrajectoryDataset, generate_synthetic_trajectories, make_dataloader` | Re-export |

**Exports:**
- Re-exports: `TrajectoryDataset`, `generate_synthetic_trajectories`, `make_dataloader`

---

### `src/pits_mras/data/trajectory.py` - Reusable trajectory dataset + synthetic generator + loader (Gap G7).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch` | `Tensor` |
| `torch.utils.data` | `DataLoader, Dataset` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |
| `typing` | `Dict, List, Sequence, Tuple, Union` |

**Exports:**
- Classes: `TrajectoryDataset`
- Functions: `generate_synthetic_trajectories`, `make_dataloader`

---

## Src / pits_mras / inference Dependencies

### `src/pits_mras/inference/__init__.py` - Inference subpackage: real-time engine and parallel thread architecture.

---

### `src/pits_mras/inference/parallel.py` - Parallel thread architecture for deployment (IP §9.2).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `copy` | `(module)` |
| `threading` | `(module)` |
| `time` | `(module)` |
| `collections` | `deque` |
| `dataclasses` | `dataclass` |
| `typing` | `Callable, Deque, List, Optional, Tuple` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/inference/realtime.py` | `RealtimeInferenceEngine` | Import |
| `src/pits_mras/losses/irl.py` | `IRLBellmanLoss` | Import |

**Exports:**
- Classes: `ControllerState`, `ParallelInferenceEngine`

---

### `src/pits_mras/inference/realtime.py` - Real-time closed-loop inference engine (IP §9.1).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `threading` | `(module)` |
| `collections` | `deque` |
| `typing` | `TYPE_CHECKING, Any, Deque, Dict, Optional` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/controllers/mras.py` | `MRASController` | Import |
| `src/pits_mras/controllers/reference_models.py` | `LinearReferenceModel` | Import |
| `src/pits_mras/models/pitnn.py` | `PITNN` | Import |
| `src/pits_mras/models/pcml.py` | `PCMLModule` | Import (TYPE_CHECKING) |

**Exports:**
- Classes: `RealtimeInferenceEngine`

---

## Src / pits_mras / losses Dependencies

### `src/pits_mras/losses/__init__.py` - Loss functions for PITS-MRAS (Phase 3).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn` | `(module)` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/config.py` | `LossConfig` | Re-export |
| `src/pits_mras/losses/hjb.py` | `HJBResidualLoss, LyapunovDecreaseEnforcer` | Re-export |
| `src/pits_mras/losses/irl.py` | `IRLBellmanAccumulator, IRLBellmanLoss` | Re-export |
| `src/pits_mras/losses/physics.py` | `PhysicsLoss` | Re-export |
| `src/pits_mras/losses/stability.py` | `ControlEffortLoss, LyapunovConstraintLoss, MRASStabilityLoss, ParameterBoundednessLoss` | Re-export |
| `src/pits_mras/losses/temporal.py` | `AttentionRegularizationLoss, MultiStepPredictionLoss, TemporalLoss, TemporalSmoothnessLoss` | Re-export |

**Exports:**
- Classes: `TotalLoss`
- Re-exports: `LossConfig`, `HJBResidualLoss`, `LyapunovDecreaseEnforcer`, `IRLBellmanAccumulator`, `IRLBellmanLoss`, `PhysicsLoss`, `ControlEffortLoss`, `LyapunovConstraintLoss`, `MRASStabilityLoss`, `ParameterBoundednessLoss`, `AttentionRegularizationLoss`, `MultiStepPredictionLoss`, `TemporalLoss`, `TemporalSmoothnessLoss`

---

### `src/pits_mras/losses/adaptive_weighting.py` - Adaptive / causal loss weighting utilities (ROADMAP #8).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |
| `typing` | `Sequence` |

**Exports:**
- Classes: `ReLoBRaLo`
- Functions: `causal_weights`

---

### `src/pits_mras/losses/hjb.py` - HJB residual loss (Phase 3).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn` | `(module)` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/models/critic.py` | `QuadraticCritic` | Import |

**Exports:**
- Classes: `HJBResidualLoss`, `LyapunovDecreaseEnforcer`

---

### `src/pits_mras/losses/irl.py` - Inverse-RL Bellman residual loss (Phase 3).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn` | `(module)` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/models/critic.py` | `QuadraticCritic` | Import |

**Exports:**
- Classes: `IRLBellmanAccumulator`, `IRLBellmanLoss`

---

### `src/pits_mras/losses/physics.py` - Physics-informed loss (Phase 3).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn` | `(module)` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |

**Exports:**
- Classes: `PhysicsLoss`

---

### `src/pits_mras/losses/stability.py` - Stability-constraint losses (Phase 3).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn` | `(module)` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |
| `typing` | `Iterable` |

**Exports:**
- Classes: `LyapunovConstraintLoss`, `ParameterBoundednessLoss`, `ControlEffortLoss`, `MRASStabilityLoss`

---

### `src/pits_mras/losses/temporal.py` - Temporal / multi-step prediction loss (Phase 3).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn` | `(module)` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |

**Exports:**
- Classes: `MultiStepPredictionLoss`, `TemporalSmoothnessLoss`, `AttentionRegularizationLoss`, `TemporalLoss`

---

## Src / pits_mras / models Dependencies

### `src/pits_mras/models/__init__.py` - Models subpackage: attention, port-Hamiltonian decoders, critic/costate, PITNN.

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/models/adversary.py` | `NeuralAdversary` | Re-export |
| `src/pits_mras/models/attention.py` | `PhysicsInformedAttention` | Re-export |
| `src/pits_mras/models/critic.py` | `AdversaryHead, CostateHead, QuadraticCritic` | Re-export |
| `src/pits_mras/models/decoders.py` | `DissipationNet, HamiltonianNet, PortHamiltonianDecoder` | Re-export |
| `src/pits_mras/models/generic.py` | `GFINNDecoder` | Re-export |
| `src/pits_mras/models/koopman.py` | `KoopmanLiftingModel, koopman_loss` | Re-export |
| `src/pits_mras/models/pitnn.py` | `PITNN` | Re-export |
| `src/pits_mras/models/sac.py` | `GaussianPolicy, TwinQCritic` | Re-export |
| `src/pits_mras/models/tdmpc.py` | `LatentModel, MPPIPlanner, WorldModel` | Re-export |

**Exports:**
- Re-exports: `NeuralAdversary`, `PhysicsInformedAttention`, `AdversaryHead`, `CostateHead`, `QuadraticCritic`, `DissipationNet`, `HamiltonianNet`, `PortHamiltonianDecoder`, `GFINNDecoder`, `KoopmanLiftingModel`, `koopman_loss`, `PITNN`, `GaussianPolicy`, `TwinQCritic`, `LatentModel`, `MPPIPlanner`, `WorldModel`

---

### `src/pits_mras/models/adversary.py` - Learned (neural) H-infinity adversary -- the disturbance policy (ROADMAP #1).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |

**Exports:**
- Classes: `NeuralAdversary`

---

### `src/pits_mras/models/attention.py` - Multi-head physics-informed attention (IP §5.1).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn` | `(module)` |
| `torch.nn.functional` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `math` | `(module)` |

**Exports:**
- Classes: `PhysicsInformedAttention`

---

### `src/pits_mras/models/critic.py` - Critic / value network and costate head (IP §5.3). NEW -- Identity 1 & 2.

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn` | `(module)` |
| `torch` | `Tensor` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/utils/lyapunov.py` | `pack_symmetric, quadratic_basis, unpack_symmetric` | Import |

**Exports:**
- Classes: `QuadraticCritic`, `CostateHead`, `AdversaryHead`

---

### `src/pits_mras/models/decoders.py` - Port-Hamiltonian decoders (IP §5.2).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn` | `(module)` |
| `torch.nn.functional` | `(module)` |
| `torch` | `Tensor` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/utils/hamiltonian.py` | `hamiltonian_positivity_loss, make_positive_definite, make_skew_symmetric, port_hamiltonian_energy_loss` | Import |

**Exports:**
- Classes: `HamiltonianNet`, `DissipationNet`, `PortHamiltonianDecoder`

---

### `src/pits_mras/models/generic.py` - GENERIC / GFINN thermodynamic decoder (ROADMAP extension).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn` | `(module)` |
| `torch` | `Tensor` |

**Exports:**
- Classes: `GFINNDecoder`

---

### `src/pits_mras/models/koopman.py` - Deep Koopman lifting model (ROADMAP proposal #2). NEW capability.

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |
| `typing` | `Dict, Tuple` |

**Exports:**
- Classes: `KoopmanLiftingModel`
- Functions: `koopman_loss`

---

### `src/pits_mras/models/lagrangian_head.py` - Lagrangian-multiplier head for the KKT projection (PCML Addendum §2.3).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn` | `(module)` |
| `torch` | `Tensor` |

**Exports:**
- Classes: `LagrangianMultiplierHead`

---

### `src/pits_mras/models/pcml.py` - Physics-Constrained Machine Learning (PCML) module for PITS-MRAS.

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn` | `(module)` |
| `torch.nn.functional` | `(module)` |
| `torch` | `Tensor` |
| `torch.func` | `jacrev, vmap` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `logging` | `(module)` |
| `typing` | `Dict, List, Optional, Tuple` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/constraints/base.py` | `PhysicsConstraints` | Import |

**Exports:**
- Classes: `SoftPCMLLoss`, `TaylorNeighborhoodApproximation`, `KKTProjectionLayer`, `PCMLModule`

---

### `src/pits_mras/models/pitnn.py` - Top-level physics-informed time-series network -- PITNN (IP §5.4).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `typing` | `Dict, Optional` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/config.py` | `NetworkConfig, PhysicsConfig` | Import |
| `src/pits_mras/models/attention.py` | `PhysicsInformedAttention` | Import |
| `src/pits_mras/models/decoders.py` | `PortHamiltonianDecoder` | Import |

**Exports:**
- Classes: `PITNN`

---

### `src/pits_mras/models/sac.py` - Soft Actor-Critic networks: squashed Gaussian policy + twin Q critic.

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |

**Exports:**
- Classes: `GaussianPolicy`, `TwinQCritic`

---

### `src/pits_mras/models/tdmpc.py` - TD-MPC2 latent world model + sampling-based (MPPI) latent planner.

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |
| `typing` | `Optional, Protocol` |

**Exports:**
- Classes: `WorldModel`, `MPPIPlanner`
- Protocols/ABCs: `LatentModel`

---

## Src / pits_mras / training Dependencies

### `src/pits_mras/training/__init__.py` - Training subpackage: physics pretrain, IRL co-train, offline IRL trainer.

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/training/cotrain.py` | `cotraining_loop` | Re-export |
| `src/pits_mras/training/hinf_minmax.py` | `hinf_minmax_from_dynamics, hinf_minmax_from_pitnn, hinf_minmax_train, hji_residual, pitnn_one_step` | Re-export |
| `src/pits_mras/training/irl_trainer.py` | `train_irl_critic` | Re-export |
| `src/pits_mras/training/pretrain.py` | `pretrain_pitnn` | Re-export |
| `src/pits_mras/training/sac.py` | `SACTrainer` | Re-export |
| `src/pits_mras/training/tdmpc.py` | `tdmpc_update` | Re-export |

**Exports:**
- Re-exports: `cotraining_loop`, `hinf_minmax_from_dynamics`, `hinf_minmax_from_pitnn`, `hinf_minmax_train`, `hji_residual`, `pitnn_one_step`, `train_irl_critic`, `pretrain_pitnn`, `SACTrainer`, `tdmpc_update`

---

### `src/pits_mras/training/cotrain.py` - Co-training pipeline with IRL critic updates (Algorithm 3 extended, §8.2).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |
| `logging` | `(module)` |
| `collections` | `deque` |
| `typing` | `TYPE_CHECKING, Deque` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/losses/adaptive_weighting.py` | `ReLoBRaLo` | Import |
| `src/pits_mras/losses/hjb.py` | `HJBResidualLoss` | Import |
| `src/pits_mras/losses/irl.py` | `IRLBellmanLoss` | Import |
| `src/pits_mras/config.py` | `LossConfig, PITSMRASConfig` | Import (TYPE_CHECKING) |
| `src/pits_mras/controllers/mras.py` | `MRASController` | Import (TYPE_CHECKING) |
| `src/pits_mras/controllers/reference_models.py` | `LinearReferenceModel` | Import (TYPE_CHECKING) |
| `src/pits_mras/controllers/safety.py` | `CLFCBFSafetyFilter` | Import (TYPE_CHECKING) |
| `src/pits_mras/models/__init__.py` | `PITNN` | Import (TYPE_CHECKING) |
| `src/pits_mras/models/critic.py` | `QuadraticCritic` | Import (TYPE_CHECKING) |
| `src/pits_mras/models/pcml.py` | `PCMLModule` | Import (TYPE_CHECKING) |

**Exports:**
- Functions: `cotraining_loop`

---

### `src/pits_mras/training/hinf_minmax.py` - Neural H-infinity adversarial min-max training loop (ROADMAP #1, capstone).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `numpy` | `(module)` |
| `torch` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |
| `logging` | `(module)` |
| `typing` | `Callable, Literal, Optional` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/models/adversary.py` | `NeuralAdversary` | Import |
| `src/pits_mras/models/critic.py` | `CostateHead, QuadraticCritic` | Import |
| `src/pits_mras/models/pitnn.py` | `PITNN` | Import |
| `src/pits_mras/utils/linearization.py` | `linearize_dynamics` | Import |
| `src/pits_mras/utils/lyapunov.py` | `solve_gare` | Import |

**Exports:**
- Functions: `hji_residual`, `hinf_minmax_train`, `hinf_minmax_from_dynamics`, `pitnn_one_step`, `hinf_minmax_from_pitnn`

---

### `src/pits_mras/training/irl_trainer.py` - Offline IRL critic trainer (batch least-squares, §8.3).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |
| `logging` | `(module)` |
| `typing` | `TYPE_CHECKING` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/utils/lyapunov.py` | `quadratic_basis, unpack_symmetric` | Import |
| `src/pits_mras/controllers/reference_models.py` | `LinearReferenceModel` | Import (TYPE_CHECKING) |
| `src/pits_mras/models/critic.py` | `QuadraticCritic` | Import (TYPE_CHECKING) |
| `src/pits_mras/losses/irl.py` | `IRLBellmanLoss` | Import |

**Exports:**
- Functions: `train_irl_critic_gd`, `train_irl_critic`

---

### `src/pits_mras/training/pretrain.py` - Physics-informed pre-training curriculum (IP §8.1, Algorithm 2).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |
| `logging` | `(module)` |
| `math` | `(module)` |
| `typing` | `TYPE_CHECKING, Callable, Iterator` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/config.py` | `PITSMRASConfig` | Import (TYPE_CHECKING) |
| `src/pits_mras/data/__init__.py` | `TrajectoryDataset` | Import (TYPE_CHECKING) |
| `src/pits_mras/models/__init__.py` | `PITNN` | Import (TYPE_CHECKING) |
| `src/pits_mras/data/__init__.py` | `make_dataloader` | Import |

**Exports:**
- Functions: `data_weight_schedule`, `temporal_weight_schedule`, `pretrain_pitnn`

---

### `src/pits_mras/training/sac.py` - Soft Actor-Critic update logic with automatic temperature (Connection 5).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn.functional` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |
| `typing` | `Mapping, Optional` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/models/sac.py` | `GaussianPolicy, TwinQCritic` | Import |

**Exports:**
- Classes: `SACTrainer`

---

### `src/pits_mras/training/tdmpc.py` - TD-MPC2 world-model update (Connection 9 — learned-model planning).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn.functional` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |
| `typing` | `Mapping` |

**Internal Dependencies:**
| Module | Imports | Type |
|--------|---------|------|
| `src/pits_mras/models/tdmpc.py` | `WorldModel` | Import |

**Exports:**
- Functions: `tdmpc_update`

---

## Src / pits_mras / utils Dependencies

### `src/pits_mras/utils/__init__.py` - Utils subpackage: Lyapunov/Riccati engine, port-Hamiltonian, PE monitor.

---

### `src/pits_mras/utils/diagnostics.py` - Long-horizon rollout-stability + conservation-drift diagnostics (ROADMAP #4).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `typing` | `Callable` |

**Exports:**
- Functions: `energy_drift`, `max_energy_drift`, `valid_prediction_time`, `rollout_jacobian_spectral_radius`

---

### `src/pits_mras/utils/hamiltonian.py` - Port-Hamiltonian utilities (IP §4.4).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch.nn.functional` | `(module)` |
| `torch` | `Tensor` |

**Exports:**
- Functions: `make_skew_symmetric`, `make_positive_definite`, `port_hamiltonian_energy_loss`, `hamiltonian_positivity_loss`

---

### `src/pits_mras/utils/linearization.py` - First-order linearization of a dynamics callable (ROADMAP integration #6).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch` | `Tensor` |
| `torch.func` | `jacrev` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |
| `typing` | `Callable, Literal, Tuple` |

**Exports:**
- Functions: `linearize_dynamics`

---

### `src/pits_mras/utils/lyapunov.py` - Lyapunov / Riccati engine (IP §4.3). "The mathematical engine for all P."

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `numpy` | `(module)` |
| `torch` | `(module)` |
| `scipy.linalg` | `schur, solve_continuous_are, solve_continuous_lyapunov` |
| `torch` | `Tensor` |
| `jaxtyping` | `Float` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `__future__` | `annotations` |
| `functools` | `lru_cache` |
| `typing` | `TYPE_CHECKING, Optional, Tuple, Union` |

**Exports:**
- Functions: `solve_lyapunov`, `kleinman_iteration`, `solve_care`, `solve_gare`, `check_hurwitz`, `differentiable_care`, `differentiable_gare`, `lyapunov_derivative`, `quadratic_basis`, `pack_symmetric`, `unpack_symmetric`

---

### `src/pits_mras/utils/pe_monitor.py` - Persistence-of-excitation monitor (IP §4.5).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `collections` | `deque` |
| `typing` | `Deque` |

**Exports:**
- Classes: `PEMonitor`

---

### `src/pits_mras/utils/uq.py` - Uncertainty quantification utilities (ROADMAP proposal #3).

**Third-party Dependencies:**
| Package | Import |
|---------|--------|
| `torch` | `(module)` |
| `torch` | `Tensor` |

**Standard-library Dependencies:**
| Module | Import |
|--------|--------|
| `math` | `(module)` |
| `typing` | `Callable, Sequence` |

**Exports:**
- Classes: `DeepEnsemble`, `AdaptiveConformalInference`
- Functions: `split_conformal_quantile`, `conformal_interval`

---

## Circular Dependency Analysis

**No circular dependencies detected.**
---

## Visual Dependency Graph

```mermaid
graph TD
    subgraph Examples
        N0[autonomous_vehicle]
        N1[building_hvac]
        N2[pcml_heat_diffusion]
        N3[plants]
        N4[robotic_manipulator]
    end

    subgraph Root
        N5[setup]
    end

    subgraph Src / pits_mras
        N6[__init__]
        N7[config]
    end

    subgraph Src / pits_mras / constraints
        N8[__init__]
        N9[base]
        N10[mechanical]
        N11[thermal]
    end

    subgraph Src / pits_mras / controllers
        N12[__init__]
        N13[koopman_control]
        N14[mras]
        N15[reference_models]
        N16[safety]
    end

    subgraph Src / pits_mras / data
        N17[__init__]
        N18[trajectory]
    end

    subgraph Src / pits_mras / inference
        N19[__init__]
        N20[parallel]
        N21[realtime]
    end

    subgraph Src / pits_mras / losses
        N22[__init__]
        N23[adaptive_weighting]
        N24[hjb]
        N25[irl]
        N26[physics]
        N27[...2 more]
    end

    subgraph Src / pits_mras / models
        N28[__init__]
        N29[adversary]
        N30[attention]
        N31[critic]
        N32[decoders]
        N33[...7 more]
    end

    subgraph Src / pits_mras / training
        N34[__init__]
        N35[cotrain]
        N36[hinf_minmax]
        N37[irl_trainer]
        N38[pretrain]
        N39[...2 more]
    end

    subgraph Src / pits_mras / utils
        N40[__init__]
        N41[diagnostics]
        N42[hamiltonian]
        N43[linearization]
        N44[lyapunov]
        N45[...2 more]
    end

    N0 --> N7
    N0 --> N14
    N0 --> N15
    N0 --> N21
    N0 --> N28
    N1 --> N7
    N1 --> N14
    N1 --> N15
    N1 --> N21
    N1 --> N28
    N2 --> N8
    N4 --> N7
    N4 --> N14
    N4 --> N15
    N4 --> N21
    N4 --> N28
    N4 --> N37
    N6 --> N8
    N6 --> N14
    N6 --> N15
    N6 --> N16
    N6 --> N21
    N6 --> N31
    N6 --> N34
    N8 --> N9
    N8 --> N10
    N8 --> N11
    N10 --> N9
    N11 --> N9
    N13 --> N44
```

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Total Python Files | 55 |
| Total Modules | 11 |
| Total Lines of Code | 9253 |
| Total Public Exports | 174 |
| Total Re-exports | 65 |
| Total Classes | 58 |
| Total Protocols/ABCs | 2 |
| Total Enums | 0 |
| Total Functions | 49 |
| Total Type Guards (is_*) | 0 |
| Total Constants | 0 |
| TYPE_CHECKING Imports | 13 |
| Runtime Circular Deps | 0 |
| TYPE_CHECKING Circular Deps | 0 |
| Potentially Unused Files | 0 |
| Potentially Unused Exports | 0 |
