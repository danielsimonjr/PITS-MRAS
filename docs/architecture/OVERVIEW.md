# PITS-MRAS — Architecture Overview

> **Orientation document.** A grounded, high-level map of the `pits_mras`
> codebase for someone reading it for the first time. Every count and module
> claim here is taken from the generated dependency graph
> (`docs/architecture/dependency-graph.json`) and the subpackage docstrings in the
> source tree. For the deep design rationale and the source-document citations, see
> `ARCHITECTURE.md` in this folder.

PITS-MRAS (**Physics-Informed Time-Series Model-Reference Adaptive Systems**) is a
research framework that merges three paradigms — **Physics-Informed Neural
Networks**, **Time-Series Deep Learning**, and **Model-Reference Adaptive Systems
(MRAS)** — and unifies them formally with reinforcement-learning / optimal-control
theory through roughly ten mathematical identities. The headline identity is that
the MRAS Lyapunov function `V(e) = eᵀPe` *is* the LQR value function for the
tracking-error system; from that, the costate `λ = ∇V`, an Integral-RL critic, and
a CLF-CBF-QP safety filter follow with near-zero extra implementation cost. The
framework includes a **PCML (Physics-Constrained ML)** layer that upgrades physics
enforcement from soft penalties to hard KKT-projection constraint satisfaction.

**55 source files · 11 modules · 9,022 LOC · 170 exports
(63 re-exports) · 58 classes · 2 Protocol/ABC · 47 functions · 0 circular
dependencies · 0 unused files/exports.**
*(Source: `dependency-graph.json` → `metadata` + `statistics`.)*

---

## 1. The Three-Paradigm Merger

PITS-MRAS combines three traditionally separate fields into a single closed-loop
adaptive controller (`ARCHITECTURE.md` §1.1):

1. **Physics-Informed Neural Networks (PINNs)** — the network's decoder enforces
   conservation laws through **port-Hamiltonian structure**: a learned Hamiltonian
   `H_θ`, a skew-symmetric interconnection `J = −Jᵀ`, and a positive-definite
   dissipation `R_θ = LᵀL`. (`models/decoders.py`, `utils/hamiltonian.py`.)
2. **Time-Series Deep Learning** — a **causal LSTM encoder** plus **multi-head
   physics-informed attention** (temporal + physical + error-driven, combined by a
   learned 3-way gate) processes the state/control/error history.
   (`models/attention.py`, `models/pitnn.py`.)
3. **Model-Reference Adaptive Systems (MRAS)** — a linear reference model
   `ẋ_m = A_m x_m + B_m r` (with Hurwitz `A_m`) defines the desired closed-loop
   behavior, and a Lyapunov function guarantees tracking-error convergence.
   (`controllers/reference_models.py`, `controllers/mras.py`.)

### The unifying idea

The framework's distinguishing move is to treat these three paradigms as facets of
one optimal-control problem, tied together by a family of **mathematical
identities** (enumerated and cited in `ARCHITECTURE.md` §3). The load-bearing
ones, as tagged in the source code itself:

| Identity | Statement (abridged) | Where it lives in code |
|---|---|---|
| **1 — Lyapunov = Value Function** | `A_mᵀP + PA_m = −Q` is Kleinman's policy-evaluation step; Integral-RL makes it model-free. | `utils/lyapunov.py`, `models/critic.py`, `losses/irl.py`, `training/irl_trainer.py` |
| **2 — Costate = Critic Gradient** | The PMP costate `λ = ∂V/∂e`; optimal control `u* = −R⁻¹Bᵀ∇V̂` is the autodiff gradient of the scalar value head. | `models/critic.py` (`CostateHead`), used by `controllers/mras.py` |
| **3 — CLF-CBF-QP Safety Filter** | One `P` matrix certifies both stability (CLF `V = eᵀPe`) and safety (CBF `h = c − eᵀPe`); closed-form single-constraint projection. | `controllers/safety.py` |
| **4 — Updated Adaptation Law (DPG)** | The MRAS update *is* a deterministic policy gradient; the learned critic gradient replaces the fixed `eᵀPe` surrogate. | `controllers/mras.py`, `training/cotrain.py` |
| **8 — HJB Residual Loss** | Adds the Hamilton-Jacobi-Bellman residual as a PINN-style regularizer. | `losses/hjb.py` |

The port-Hamiltonian storage function doubling as a value function (passivity =
L2-gain) and the neural/generalized-Lyapunov residual are additional rigorous
connections. The remaining Blueprint connections are also built out as dedicated
modules: the H∞ adversary head (analytic `AdversaryHead` + neural `NeuralAdversary`
with its min-max trainer), SAC-entropy (`models/sac.py` + `training/sac.py`), and
TD-MPC2 latent planning (`models/tdmpc.py` world model + MPPI planner). See
`ARCHITECTURE.md` §3 for the full identity → module mapping.

### The PCML layer

**Physics-Constrained ML** upgrades physics enforcement from *soft* penalties to
*hard* constraint satisfaction
(`docs/PITS-MRAS_FINAL_SUMMARY.md`, PCML section):

- **Soft PCML** (pre-training): augments the loss with DAE residuals
  `λ_diff‖D‖² + λ_eq‖h‖² + λ_ineq‖ReLU(g)‖²`.
- **Hard PCML** (co-training + inference): a differentiable **KKT-projection**
  layer projects predicted dynamics onto the differential-algebraic constraint
  manifold to machine precision (Newton on the KKT system with Fischer-Burmeister
  complementarity; gradients via a one-step implicit-function trick).

PCML is **opt-in** — all PCML config flags default off, so the base PITNN behavior
applies unless PCML is explicitly wired in.

---

## 2. Module Map

Eleven modules, 55 files (incl. the `examples/`). One-line purposes are taken verbatim (abridged) from each
subpackage's `__init__.py` docstring as recorded in the dependency graph.

| Module | Files | Purpose |
|---|---|---|
| `src/pits_mras` (package root) | 2 | Top-level package: `config.py` (centralized dataclass config, IP §4.2) + `__init__.py` (public-API barrel re-exporting 17 symbols). |
| `src/pits_mras/constraints` | 4 | Physics constraint systems for PCML (PCML Addendum §2.1): `PhysicsConstraints` ABC, `MechanicalDAE`, `HeatConductionDAE`. |
| `src/pits_mras/controllers` | 5 | Reference models, CLF-CBF safety filter, the actor-critic MRAS controller, and the Koopman-LQR controller (CARE on lifted coords). |
| `src/pits_mras/data` | 2 | Opt-in trajectory dataset/loader: `TrajectoryDataset`, `generate_synthetic_trajectories`, `make_dataloader`. |
| `src/pits_mras/inference` | 3 | Real-time closed-loop inference engine and the parallel multi-thread deployment architecture (IP §9). |
| `src/pits_mras/losses` | 7 | Loss functions (Phase 3): physics, temporal, stability, IRL-Bellman, HJB-residual, adaptive/causal weighting, plus the `TotalLoss` aggregator. |
| `src/pits_mras/models` | 12 | Physics-informed attention, port-Hamiltonian decoders, critic/costate/adversary heads, PCML/Lagrangian heads, neural adversary, deep Koopman lifting, SAC policy/critic nets, TD-MPC2 world model + MPPI planner, the GENERIC/GFINN thermodynamic decoder, and the top-level `PITNN`. |
| `src/pits_mras/training` | 7 | Physics pre-training curriculum, IRL co-training loop, the offline IRL critic trainer (IP §8), the H∞ neural min-max loop, the SAC trainer, and the TD-MPC2 world-model update. |
| `src/pits_mras/utils` | 7 | Lyapunov/Riccati engine (+ GARE + differentiable CARE/GARE), port-Hamiltonian utilities, the PE monitor, rollout diagnostics, UQ utilities, and dynamics linearization. |
| `examples` | 5 | Runnable end-to-end demos: robotic manipulator, autonomous vehicle, building HVAC, hard-PCML heat-diffusion + `plants.py` nonlinear plant steps. |
| `root` | 1 | `setup.py` packaging entry point. |

### Notable files within each subpackage

- **`utils/lyapunov.py`** — "the mathematical engine for all P": `solve_lyapunov`,
  `kleinman_iteration`, `solve_care`, `check_hurwitz`, `lyapunov_derivative`,
  `quadratic_basis`, and the canonical `pack_symmetric`/`unpack_symmetric`
  basis helpers (built on `scipy.linalg`).
- **`models/critic.py`** — `QuadraticCritic` (`V̂ = Wᵀφ(e)`), `CostateHead`
  (`λ̂ = ∇V̂`, `u* = −R⁻¹Bᵀλ̂`), and `AdversaryHead` (H∞ worst-case `w* = γ⁻²DᵀPe`);
  tagged "Identity 1 & 2."
- **`models/pcml.py`** — `SoftPCMLLoss`, `TaylorNeighborhoodApproximation`,
  `KKTProjectionLayer`, `PCMLModule` (the soft/hard mode manager).
- **`controllers/mras.py`** — `MRASController`, combining classical MRAS
  feedforward/feedback with the critic, costate head, and CBF filter (Identities
  1–4).
- **`inference/realtime.py`** — `RealtimeInferenceEngine`; `inference/parallel.py`
  adds `ParallelInferenceEngine` + `ControllerState` for multi-rate threaded
  deployment.

---

## 3. Entry Points

The graph lists **13 entry points**: the 4 example scripts (CLI), the package +
6 subpackage `__init__.py` barrels, plus `__init__.py` at the root and `setup.py`.

### 3.1 Examples (CLI)

Each example is a self-contained script exporting `run` and `main`. They depend on
the public surface — `config` (`NetworkConfig`, `PhysicsConfig`,
`PITSMRASConfig`), `MRASController`, `LinearReferenceModel`,
`RealtimeInferenceEngine`, and `PITNN` — and use `numpy`, `torch`, and
`matplotlib`.

| Script | Scenario | Design ref |
|---|---|---|
| `examples/robotic_manipulator.py` | 2-DOF planar robotic manipulator | IP §10.1 |
| `examples/autonomous_vehicle.py` | Autonomous-vehicle lateral control | IP §10.2 |
| `examples/building_hvac.py` | Building HVAC thermal-zone control | IP §10.3 |
| `examples/pcml_heat_diffusion.py` | Hard PCML on the 1-D heat equation (real `(x,t,∂)`) | PCML / DAE-HardNet |

Run an example directly with Python, e.g.:

```bash
python examples/robotic_manipulator.py
```

### 3.2 Public API (`import pits_mras`)

The package `__init__.py` re-exports **17** symbols (its `__all__`), spanning the
core model/control stack plus the PCML layer:

- **Core:** `PITNN`, `QuadraticCritic`, `MRASController`, `LinearReferenceModel`,
  `CLFCBFSafetyFilter`, `RealtimeInferenceEngine`.
- **Training functions:** `pretrain_pitnn`, `cotraining_loop`.
- **PCML:** `PhysicsConstraints`, `ConstraintSpec`, `MechanicalDAE`,
  `HeatConductionDAE`, `SoftPCMLLoss`, `TaylorNeighborhoodApproximation`,
  `KKTProjectionLayer`, `PCMLModule`, `LagrangianMultiplierHead`.

```python
import pits_mras
from pits_mras import PITNN, MRASController, RealtimeInferenceEngine
```

---

## 4. Key Statistics

All values from `dependency-graph.json` → `statistics` (and `metadata`).

| Metric | Value |
|---|---|
| Total Python files | 55 |
| Modules | 11 |
| Total lines of code | 9,022 |
| Total exports | 170 |
| Re-exports (barrel) | 63 |
| Classes | 58 |
| Interfaces (Protocol/ABC) | 2 |
| Enums | 0 |
| Functions | 47 |
| Constants | 0 |
| Type-checking-only imports | 10 |
| Runtime circular dependencies | 0 |
| Type-only circular dependencies | 0 |
| Unused files | 0 |
| Unused exports | 0 |

The single interface/ABC is `PhysicsConstraints` in `constraints/base.py`. The
63 re-exports are the convenience barrels in the package and subpackage
`__init__.py` files (`pits_mras`, `constraints`, `losses`, `models`, `training`).
The 10 type-only imports are `TYPE_CHECKING`-guarded edges (mostly into
`training/cotrain.py`, `training/irl_trainer.py`, `training/pretrain.py`, and
`inference/realtime.py`) that keep import-time dependencies light.

### Dependency layers

The graph organizes the codebase into clean layers (`dependencyGraph.layers`):
Examples → Root → package root (`pits_mras`) → `constraints` → `controllers` →
`inference` → `losses` → `models` → `training` → `utils`. Internal runtime
dependencies flow downward (e.g. `controllers/mras.py` → `models/critic.py` and
`utils/lyapunov.py`; `models/decoders.py` → `utils/hamiltonian.py`), with `utils`
and `config` as the leaf foundation that depends on nothing internal.

---

## 5. Codebase Health

The generated graph reports a clean structure:

- **0 circular dependencies** — `runtimeCount: 0` and `typeOnlyCount: 0`
  (`dependencyGraph.circularDependencies`). The dependency graph is a strict DAG.
- **0 unused files** — `unusedFilesCount: 0` (`statistics`).
- **0 unused exports** — `unusedExportsCount: 0` (`statistics`); see
  `unused-analysis.md` for the per-symbol breakdown.

Every exported symbol is reachable from an entry point, and no module forms an
import cycle with another. The use of `TYPE_CHECKING` guards in the training and
inference modules keeps cross-module type references from introducing runtime
cycles.

---

## 6. Where to Go Next

Other documents in `docs/architecture/`:

- **`ARCHITECTURE.md`** — the full engineering blueprint (with a §0 graph-backed
  as-built summary): the three-paradigm merger in depth, the identity → module
  mapping (all ten connections), data-flow diagrams, the training/inference
  pipelines, the stability/safety/testing strategy, dependencies, and the flagged
  open gaps. This is the authoritative design reference.
- **`COMPONENTS.md`** — per-module component breakdown: each module's files, key
  classes/functions, responsibilities, and internal dependencies.
- **`API.md`** — public API reference: the 17 top-level symbols, their
  constructor/method signatures and return shapes, the `PITSMRASConfig`
  dataclasses, and an end-to-end usage snippet.
- **`DATAFLOW.md`** — the runtime data flow across pre-training, co-training, and
  real-time inference (PITNN → PCML → controller → CBF → plant), with Mermaid
  diagrams and tensor shapes.
- **`DEPENDENCY_GRAPH.md`** — the human-readable form of
  `dependency-graph.json`: per-module file listings, import edges, exports, and
  the layer ordering.
- **`TEST_COVERAGE.md`** — the test-suite map and coverage status.
- **`unused-analysis.md`** — the reachability analysis backing the
  "0 unused files / 0 unused exports" health claim.

Project-level references outside this folder:

- **`docs/PITS-MRAS_FINAL_SUMMARY.md`** — the validated feature summary, including
  the PCML section.
- **`src/pits_mras/__init__.py`** — the canonical public API surface.

> *This overview is derived from `dependency-graph.json` (the generated graph),
> the subpackage `__init__.py` docstrings, `ARCHITECTURE.md`, and
> `docs/PITS-MRAS_FINAL_SUMMARY.md`. No source code was modified in producing it.*
