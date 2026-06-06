# PITS-MRAS — Components

> A grounded, per-module component reference for the `pits_mras` package.
> Every class/function name, responsibility, and dependency below is taken from
> the actual source under `src/pits_mras/` and cross-checked against
> `docs/architecture/dependency-graph.json` (55 files, 11 modules, 170
> exports, 0 circular dependencies). Module purposes quote the package
> `__init__` and file docstrings; class/function responsibilities quote their
> own docstrings. For the *why* (the ten RL/optimal-control identities and the
> two source design documents), see `ARCHITECTURE.md`.

---

## 1. Layered Component Organization

PITS-MRAS is built bottom-up as a strict dependency stack. The dependency graph
records **zero** runtime or type-only circular dependencies; every arrow points
"downward" toward more foundational layers.

```
                ┌─────────────────────────────────────────────┐
   examples/    │ robotic_manipulator · autonomous_vehicle ·   │  runnable demos
                │ building_hvac  (→ config, controllers,        │
                │                  inference, models.PITNN)     │
                └───────────────────────┬─────────────────────┘
                                        │
   inference/   ┌───────────────────────▼─────────────────────┐
                │ RealtimeInferenceEngine → ParallelInference  │  closed-loop runtime
                │ (→ controllers, models.PITNN, models.PCML)   │
                └───────────────────────┬─────────────────────┘
                                        │
   training/    ┌───────────────────────▼─────────────────────┐
                │ pretrain_pitnn · cotraining_loop ·           │  optimization pipelines
                │ train_irl_critic  (→ losses, controllers,    │
                │                     models, config)          │
                └───────────────────────┬─────────────────────┘
                                        │
   controllers/ ┌────────────▼──────────┴──────────┐ constraints/ ┌──────────────┐
                │ MRASController                    │             │ PhysicsCon-  │
                │ LinearReferenceModel              │             │ straints     │
                │ CLFCBFSafetyFilter                │             │ Mechanical/  │
                │ (→ models.critic, utils.lyapunov) │             │ HeatDAE      │
                └────────────┬──────────────────────┘             └──────┬───────┘
                             │                                           │
   losses/      ┌────────────▼───────────────────────────────────────┐  │
                │ PhysicsLoss · TemporalLoss · MRASStabilityLoss ·     │  │
                │ IRLBellmanLoss · HJBResidualLoss · TotalLoss         │  │
                │ (→ models.critic, config.LossConfig)                 │  │
                └────────────┬─────────────────────────────────────────┘  │
                             │                                             │
   models/      ┌────────────▼─────────────────────────────────────────▼─┐
                │ PITNN ← attention + decoders ;  QuadraticCritic/Costate  │  neural networks
                │ PCMLModule/KKTProjectionLayer ; LagrangianMultiplierHead │
                │ (→ utils.hamiltonian, utils.lyapunov, constraints.base)  │
                └────────────┬─────────────────────────────────────────────┘
                             │
   utils/       ┌────────────▼─────────────────────────────────────────────┐
                │ lyapunov (solve_care/kleinman) · hamiltonian · PEMonitor   │  pure-math foundation
                └────────────────────────────────────────────────────────────┘

   config.py    PITSMRASConfig + 7 sub-configs — read by every layer above.
```

**Reading the layers (from the graph's `dependencyGraph.layers` and per-file
`internalDependencies`):**

- **`config.py`** sits beside the package root with **no internal
  dependencies** — it is a pure leaf that everything else reads.
- **`utils/`** has no internal dependencies (pure NumPy/SciPy/Torch math). It is
  the foundation: `models` and `controllers` import from it.
- **`constraints/`** depends only on its own `base.py`; `models/pcml.py` imports
  `PhysicsConstraints` from it.
- **`models/`** imports from `utils/` and `constraints/` (and `config` for
  PITNN). Nothing in `models/` imports `controllers`, `losses`, `training`, or
  `inference`.
- **`losses/`** imports `models.critic` (HJB and IRL losses) and
  `config.LossConfig`. No controller/training/inference imports.
- **`controllers/`** imports `models.critic`, `utils.lyapunov`, and (for
  `mras.py`) its own `reference_models` and `safety`.
- **`training/`** imports `losses`, plus `controllers`, `models`, and `config`
  (several as `TYPE_CHECKING`-only type hints).
- **`inference/`** imports `controllers` and `models`; `parallel.py` imports
  `realtime.py`.
- **`examples/`** sit at the top, importing `config`, `controllers`,
  `inference`, and `models.PITNN`.

The package root `src/pits_mras/__init__.py` re-exports a flat public API of 17
symbols: `PITNN`, `QuadraticCritic`, `MRASController`,
`LinearReferenceModel`, `CLFCBFSafetyFilter`, `RealtimeInferenceEngine`,
`pretrain_pitnn`, `cotraining_loop`, plus the PCML surface
(`PhysicsConstraints`, `ConstraintSpec`, `MechanicalDAE`, `HeatConductionDAE`,
`SoftPCMLLoss`, `TaylorNeighborhoodApproximation`, `KKTProjectionLayer`,
`PCMLModule`, `LagrangianMultiplierHead`).

---

## 2. Package Root — `src/pits_mras/`

**Purpose** (from `__init__.py` docstring): "A unified framework merging
Physics-Informed Neural Networks (PINNs), Time-Series Deep Learning, and
Model-Reference Adaptive Control (MRAS)." The MRAS Lyapunov function `V(e)=eᵀPe`
is the LQR value function; policy iteration on the CARE (Kleinman 1968) is the
backbone; IRL (Vrabie & Lewis 2009) makes it model-free.

| File | Key exports | Responsibility |
|---|---|---|
| `__init__.py` | the 17-symbol public API + `__version__` | Re-exports the public surface from the subpackages; the single import target for users (`import pits_mras`). |
| `config.py` | `NetworkConfig`, `PhysicsConfig`, `MRASConfig`, `SafetyConfig`, `LossConfig`, `TrainingConfig`, `PCMLConfig`, `PITSMRASConfig` | Centralized, stdlib-`dataclasses`-based configuration (not pydantic). |

### `config.py` — the eight dataclasses

| Dataclass | Holds |
|---|---|
| `NetworkConfig` | PITNN architecture: `input_dim=10`, `hidden_dim=128`, `output_dim=4`, `lstm_layers=2`, `attention_heads=4`, `memory_horizon=50`, `embedding_dim=64`. |
| `PhysicsConfig` | Port-Hamiltonian decoder dims: `n_generalized_coords=2`, `hamiltonian_hidden`, `dissipation_hidden`, `use_position_dependent_J`. |
| `MRASConfig` | Classical + IRL/actor-critic params: `state_dim`, `control_dim`, reference matrices `A_m/B_m/C_m`, LQR cost `Q_cost/R_cost`, adaptation gains, `irl_window_size`, `use_irl_critic`. |
| `SafetyConfig` | CLF-CBF filter: `enable_cbf`, `safety_margin` (the `c` in `h(e)=c−eᵀPe`), `cbf_decay_rate` (the `γ`). |
| `LossConfig` | All aggregator-level loss weights: `lambda_physics/temporal/stability/data/irl/hjb/pcml` plus the physics sub-weights (`lambda_energy/pde/bc/sym`). (`lambda_hjb` defaults to `0.0` — opt-in critic regularizer; per-sub-loss weights live on the individual loss classes.) |
| `TrainingConfig` | Schedule for Algorithm 2/3: `pretrain_epochs`, stage epochs, `n_episodes`, `dt`, `device`, `seed`, logging cadence. |
| `PCMLConfig` | Physics-Constrained ML module: soft-mode residual weights, hard-mode (DAE-HardNet) params (`omega`, `eta`, `delta`, `taylor_order`, Newton settings), and constraint-system selection (`constraint_type` = `"mechanical"`/`"thermal"`, `n_joints`, thermal bounds). |
| `PITSMRASConfig` | Master config aggregating the seven sub-configs via `field(default_factory=...)`; provides `from_yaml` / `to_yaml`. The single object passed to all components. |

**Dependencies:** `config.py` has **no internal dependencies**. It is imported
by `pitnn.py`, `cotrain.py`, `pretrain.py`, `losses/__init__.py` (for
`LossConfig`), and all three examples.

---

## 3. `utils/` — Foundation math layer

**Purpose** (from `__init__.py`): "Lyapunov/Riccati engine, port-Hamiltonian, PE
monitor… pure-math utilities that the network modules will import." Phase 1.
Empty (docstring-only) package init — no re-exports; callers import the
submodules directly.

| File | Key functions / classes | Responsibility |
|---|---|---|
| `lyapunov.py` | `solve_lyapunov`, `kleinman_iteration`, `solve_care`, `solve_gare`, `check_hurwitz`, `differentiable_care`, `differentiable_gare`, `lyapunov_derivative`, `quadratic_basis`, `pack_symmetric`, `unpack_symmetric` | "The mathematical engine for all P." SciPy-backed Lyapunov/Riccati/H∞-game solvers plus their differentiable (implicit-function-theorem) variants and the symmetric-pack basis (Identity 1 foundation). |
| `hamiltonian.py` | `make_skew_symmetric`, `make_positive_definite`, `port_hamiltonian_energy_loss`, `hamiltonian_positivity_loss` | Port-Hamiltonian structure helpers (Connection 2: storage = value). |
| `pe_monitor.py` | `PEMonitor` | Persistence-of-excitation monitor for IRL/ADP convergence. |
| `diagnostics.py` | `energy_drift`, `max_energy_drift`, `valid_prediction_time`, `rollout_jacobian_spectral_radius` | Long-horizon rollout-stability + conservation-drift diagnostics (pure `torch`). |
| `uq.py` | `DeepEnsemble`, `split_conformal_quantile`, `conformal_interval`, `AdaptiveConformalInference` | Uncertainty quantification: epistemic (ensemble spread) + distribution-free conformal coverage. |
| `linearization.py` | `linearize_dynamics` | First-order (Jacobian) linearization of a dynamics callable `f(x,u)` at an operating point. |
| `__init__.py` | (none) | Docstring-only package init. |

**`lyapunov.py` function detail:**
- `solve_lyapunov(A_m, Q)` — solves `A_mᵀP + PA_m = −Q` for `P≻0`; raises `ValueError` if `P` is not positive definite (i.e. `A_m` not Hurwitz).
- `kleinman_iteration(A,B,Q,R,…)` — Kleinman's 1968 policy iteration: alternates policy evaluation (closed-loop Lyapunov solve) and policy improvement `K ← R⁻¹BᵀP`; returns `(P_star, K_star)`.
- `solve_care(A,B,Q,R)` — solves the CARE directly via `scipy.linalg.solve_continuous_are`; returns `(P_star, K_star)`.
- `solve_gare(A,B,Q,R,gamma,D=None)` — solves the H∞ Game ARE `AᵀP+PA+Q−P(BR⁻¹Bᵀ−γ⁻²DDᵀ)P=0` via the Hamiltonian-Schur method (no iteration); returns `(P, K, L)` — the stabilizing `P`, robust gain `K=R⁻¹BᵀP`, and worst-case disturbance gain `L=γ⁻²DᵀP`. Raises `ValueError` when `gamma` is infeasible (no proper stable subspace / `P` not PD / worst-case loop not Hurwitz). `D` defaults to `B`.
- `check_hurwitz(A)` — `True` if all eigenvalues of `A` have negative real parts.
- `differentiable_care(A,B,Q,R)` / `differentiable_gare(A,B,Q,R,gamma,D=None)` — `torch.autograd.Function`-wrapped CARE/GARE solves: the forward solve runs under `torch.no_grad` via the scipy-backed direct solvers, while the backward differentiates the Riccati *residual* `F(P,θ)=0` implicitly so gradients flow back to `A,B,Q,R` (and `D` for the GARE). `gamma` is a non-differentiable scalar; `Q`/`R` are symmetrized internally.
- `lyapunov_derivative(e,P,A_m,B,u)` — analytic `V̇ = 2eᵀP(A_m e + Bu)`.
- `quadratic_basis(e)` — upper-triangular Kronecker basis `[e1², e1·e2, …, en²]` so a linear critic represents `V̂ = eᵀP̂e` exactly.
- `pack_symmetric(P)` / `unpack_symmetric(vec,n)` — the single source of truth for the basis convention: pack a symmetric `[n,n]` matrix into the row-major upper-triangular coefficient vector (diagonal coeff = `P[i,i]`, cross-term coeff = `P[i,j]+P[j,i]`) so `eᵀPe = φ(e)ᵀpack(P)`, and its inverse (off-diagonals split symmetrically).

**`hamiltonian.py` function detail:** `make_skew_symmetric` returns
`(raw−rawᵀ)/2` (guarantees `J=−Jᵀ`); `make_positive_definite` returns
`LᵀL + εI ≻ 0`; `port_hamiltonian_energy_loss` is the dissipation-inequality
residual `‖dH/dt − P_control + P_diss‖²`; `hamiltonian_positivity_loss` is
`mean(ReLU(−H))`.

**`PEMonitor`** tracks the min eigenvalue of the regressor Gram matrix over a
sliding window (`update`, `is_pe_satisfied`) and emits probing noise
(`get_probing_noise`) when PE is not met.

**`diagnostics.py` function detail:** `energy_drift(quantity, relative=True)`
returns the per-timestep drift of a nominally conserved series (e.g. `H(t)`)
versus its first sample — `q_t−q_0`, or `(q_t−q_0)/(|q_0|+eps)` when relative —
with `drift[...,0]==0` by construction; `max_energy_drift` is its worst-case
absolute excursion over the time axis. `valid_prediction_time(pred, truth,
threshold, dt)` returns the VPT — `t*·dt` where `t*` is the first step whose
normalized L2 error exceeds `threshold` (else the full `(T−1)·dt` horizon).
`rollout_jacobian_spectral_radius(step_fn, x)` computes the one-step Jacobian via
`torch.autograd.functional.jacobian` and returns its spectral radius `ρ(J)` — the
local error-amplification factor (`ρ>1` ⇒ geometric error growth / instability).

**`uq.py` detail:** `DeepEnsemble` wraps `K` independent member callables and
exposes `predict_all` (stacked `[K,batch,d]`) and `mean_and_std` (the population
std is the epistemic-uncertainty estimate). `split_conformal_quantile(scores,
alpha)` returns the finite-sample (`n+1`-corrected) split-conformal half-width
giving ≥`1−alpha` marginal coverage (`+inf` when `alpha` is too small for the
calibration size); `conformal_interval(pred, q)` forms the symmetric interval
`(pred−q, pred+q)`. `AdaptiveConformalInference` is the online (Gibbs & Candès)
update `α_{t+1}=α_t+γ(α*−err_t)` that tracks a running miscoverage level
(`current_alpha`, `update(covered)`), clamped to `[0,1]`.

**`linearization.py` detail:** `linearize_dynamics(dynamics_fn, x0, u0)` returns
the tangent linear model `(A, B)` — `A=∂f/∂x`, `B=∂f/∂u` at `(x0,u0)` — via
`torch.func.jacrev` (`argnums=0`/`1`), detached on the dtype/device of `x0`. It
is exact for affine `f` and a local first-order approximation otherwise; it
validates the Jacobian shapes and requires single (unbatched) `x`/`u` inputs.

**Dependencies:** no internal dependencies. **Imported by:**
`models/critic.py` (`quadratic_basis`), `models/decoders.py` (all four
hamiltonian functions), `controllers/reference_models.py`
(`solve_lyapunov`, `kleinman_iteration`, `check_hurwitz`),
`controllers/mras.py` (`solve_care`).

---

## 4. `constraints/` — Physics-constraint (DAE) systems for PCML

**Purpose** (from `__init__.py`): "Plug-in DAE constraint definitions consumed
by the soft PCML loss and the hard KKT projection layer. Each system exposes
differential / equality / inequality residuals via the `PhysicsConstraints`
interface." (PCML Addendum §2.1.)

| File | Key classes | Responsibility |
|---|---|---|
| `base.py` | `ConstraintSpec` (dataclass), `PhysicsConstraints` (ABC) | Abstract base for a plant's governing DAE: `differential(x,t,y,d)=0`, `equality(x,t,y)=0`, `inequality(x,t,y)≤0`, plus `violation`. The same residuals feed both the soft loss (squared) and the hard KKT projection (zeroed). |
| `mechanical.py` | `MechanicalDAE` | Euler-Lagrange DAE constraints with optional holonomic constraints (joint-bound inequalities). |
| `thermal.py` | `HeatConductionDAE` | 1-D transient heat conduction `dT/dt = α·d²T/dx²` with temperature-bound inequalities; derivative layout `d=[dT_dx, dT_dt, d²T_dx², d²T_dt²]`. |
| `__init__.py` | re-exports all four | Subpackage public surface. |

`ConstraintSpec` records `n_differential`, `n_equality`, `n_inequality`,
`n_outputs` so the projection layer can stay constraint-agnostic.

**Dependencies:** `mechanical.py` and `thermal.py` import `ConstraintSpec` and
`PhysicsConstraints` from `base.py`. **Imported by:** `models/pcml.py`
(`PhysicsConstraints`); the package root re-exports the four classes.

---

## 5. `models/` — Neural network components

**Purpose** (from `__init__.py`): "attention, port-Hamiltonian decoders,
critic/costate, PITNN." Phase 2. Re-exports the six core model classes;
`pcml.py` and `lagrangian_head.py` are imported directly / via the package root.

| File | Key classes | Responsibility |
|---|---|---|
| `attention.py` | `PhysicsInformedAttention` | Three-headed attention (temporal + physical + error-driven) fused by a learned 3-way softmax gate into context `c_t` and weights `alpha`. |
| `decoders.py` | `HamiltonianNet`, `DissipationNet`, `PortHamiltonianDecoder` | Port-Hamiltonian decoder stack (Connection 2). |
| `critic.py` | `QuadraticCritic`, `CostateHead`, `AdversaryHead` | Value head + costate/optimal-control head (Identity 1 & 2) + H∞ worst-case-disturbance head. |
| `pitnn.py` | `PITNN` | Top-level dynamics model (Algorithm 1): embed → causal LSTM → attention → port-Hamiltonian decoder. |
| `pcml.py` | `SoftPCMLLoss`, `TaylorNeighborhoodApproximation`, `KKTProjectionLayer`, `PCMLModule` | Physics-Constrained ML: soft penalty (Patel et al. 2022) and hard KKT projection (DAE-HardNet). |
| `lagrangian_head.py` | `LagrangianMultiplierHead` | Predicts KKT warm-start multipliers `lambda_hat` from the attention context (PCML Addendum §2.3). |
| `adversary.py` | `NeuralAdversary` | Learned worst-case-disturbance policy `w=π_w(e)` (Tanh-MLP); the independent-learner counterpart to the analytic `AdversaryHead`, trained by ascent in the min-max loop. |
| `koopman.py` | `KoopmanLiftingModel`, `koopman_loss` | Deep Koopman lifting model (Lusch et al. 2018, with control): encoder → exactly-linear latent dynamics → decoder, plus the reconstruction/linearity/prediction training losses. |
| `sac.py` | `GaussianPolicy`, `TwinQCritic` | Soft Actor-Critic networks: tanh-squashed diagonal-Gaussian stochastic actor + twin (clipped double-Q) critic. |
| `tdmpc.py` | `WorldModel`, `MPPIPlanner`, `LatentModel` | TD-MPC2 latent world model (encoder/dynamics/reward/value/Q heads) + sampling-based MPPI/CEM latent planner; `LatentModel` is the structural `Protocol` the planner consumes. |
| `generic.py` | `GFINNDecoder` | GENERIC/GFINN thermodynamic decoder enforcing skew `L`, PSD `M`, and the degeneracy conditions by construction (first + second laws hold structurally). |
| `__init__.py` | re-exports the model classes | Subpackage public surface. |

**Class detail:**
- `PhysicsInformedAttention` — temporal attention (scaled dot-product over LSTM states), physical attention (learned map over `[x_p, x_p_dot, u]`), error-driven attention (cosine similarity between current and past tracking errors).
- `HamiltonianNet` — 2-hidden-layer `Tanh` MLP with `Softplus` head so `H_θ > 0`.
- `DissipationNet` — emits a lower-triangular Cholesky factor `L`, returns `R_θ = LᵀL + εI ⪰ 0`.
- `PortHamiltonianDecoder` — `f̂ = J(q)∇H − [0; R_θ(q)·∂H/∂p] + B(x_p)u + W_corr c_t + b_corr`, with `J=−Jᵀ`, `R_θ⪰0`, `H_θ>0`; Hamiltonian gradient via autograd (`create_graph=True`). Dissipation is pH-consistent so the energy residual vanishes by construction for the conservative/dissipative/control terms.
- `QuadraticCritic` — `V̂(e)=W_cᵀφ(e)` over the upper-triangular basis, so `V̂=eᵀP̂e` with `P̂` symmetric and the LQR limit exactly representable; `W_c` initialized near `P̂≈I`; optional `nonlinear_residual` MLP for the nonlinear regime (Connection 10). No bias (`V(0)=0` required for a CLF).
- `CostateHead` — `λ̂ = ∂V̂/∂e`, `u* = −½R⁻¹Bᵀλ̂`; the action head IS the autodiff gradient of the critic (Identity 2 by construction). Documents the shared factor-of-½ convention with `HJBResidualLoss`.
- `PITNN` — sliding `(state, control)` history + current error → dynamics prediction `f̂_θ`; causal (forward-only LSTM, no future leakage), energy-conserving (port-Hamiltonian), positive dissipation. Optional `lagrangian_head` emits KKT warm-start multipliers without changing the base output contract.
- `SoftPCMLLoss` — `L = λ_diff‖D‖² + λ_eq‖h‖² + λ_ineq‖ReLU(g)‖²` (Patel et al. 2022); generalizes the port-Hamiltonian `L_physics`.
- `TaylorNeighborhoodApproximation` — multi-point neighborhood approximation converting differential operators into algebraic variables `d` for the KKT projection (DAE-HardNet Eq. 9).
- `KKTProjectionLayer` — differentiable Newton projection onto the DAE constraint manifold (min-distance problem with Fischer-Burmeister complementarity).
- `PCMLModule` — wraps a backbone prediction, returning the constrained prediction + PCML loss, switching from soft to hard mode once the data loss drops below `eta` (`update_activation`).
- `NeuralAdversary` — a plain Tanh-MLP mapping the tracking error `e` to a disturbance `w`; the output layer is linear (no squashing) so the loop can recover the linear analytic policy `w*(e)=L*e` in the LTI regime, and its output weights are initialized small so the adversary starts as a weak perturbation. Additive counterpart to `AdversaryHead` (which it leaves untouched as oracle/warm-start).
- `KoopmanLiftingModel` — learnable `encode` + exactly-linear `latent_step` `z_{k+1}=z_k A_zᵀ + u_k B_zᵀ` (no bias/nonlinearity) + `decode`. With `include_state=True` (default) the lift is `z=[x; ψ(x)]` so the decoder is the exact state slice (zero reconstruction error) and warm-starts as the trivial linear-in-state predictor; `forward` is `decode(latent_step(encode(x),u))`. `latent_matrices()` returns `(A_z, B_z)` — the bridge to the linear core (`solve_care`/`solve_gare`), which it does not call itself.
- `koopman_loss(model, x, u, x_next, …)` — the three canonical deep-Koopman MSE terms `recon` (reconstruction, zero when `include_state=True`), `lin` (latent linearity `‖g(x_next)−L(g(x),u)‖²`), `pred` (state prediction), plus the weighted `loss`.
- `GaussianPolicy` — a shared MLP trunk emitting per-dim `mean`/`log_std` (clamped); `sample` draws a reparameterized `a=scale·tanh(mean+std·ε)` and returns the exact squashed-Gaussian `log_prob` (with the tanh change-of-variables and `log(scale)` Jacobian corrections); `mean` is the deterministic greedy action.
- `TwinQCritic` — two independent Q-networks (clipped double-Q); `forward` returns `(q1,q2)` and `q_min` their elementwise minimum, mitigating value over-estimation.
- `WorldModel` — TD-MPC2 latent model as ReLU-MLP heads over a shared latent: `encode`, `next` (latent dynamics), `reward`, `value` (terminal `V`), and an optional `Q`; `forward` returns the differentiable `(z_next, reward, value(z_next))`.
- `MPPIPlanner` — gradient-free (`@torch.no_grad`) sampling-based latent MPC: samples `N` length-`H` action sequences from a per-step diagonal Gaussian, rolls each through the latent model scoring discounted reward + discounted terminal `value`, then re-fits the sampler to the top-`k` elites with MPPI exponential weighting `w_i∝exp((R_i−R_max)/temperature)`; after `iterations` refinements returns the planned first action. `LatentModel` is the `Protocol` (`next`/`reward`/`value`) it accepts.
- `GFINNDecoder` — learns scalar potentials `E(z)`, `S(z)` (autograd gradients, `create_graph=True`) and parameterizes the skew operator `L(z)=Σ(â_k b̂_kᵀ−b̂_k â_kᵀ)` (projected ⟂ `∇S`) and the PSD friction operator `M(z)=D̂D̂ᵀ` (columns projected ⟂ `∇E`); `forward(z)` returns the GENERIC field `ż=L∇E+M∇S`, so energy conservation (`∇Eᵀż=0`) and entropy production (`∇Sᵀż≥0`) hold by construction.

**Dependencies:** `attention.py`, `lagrangian_head.py`, `adversary.py`,
`sac.py`, `tdmpc.py`, and `generic.py` have no internal deps;
`koopman.py` has no internal deps (its bridge to `utils.lyapunov` is documented
but only realized by callers);
`critic.py` imports `utils.lyapunov.quadratic_basis`; `decoders.py` imports the
four `utils.hamiltonian` functions; `pcml.py` imports
`constraints.base.PhysicsConstraints`; `pitnn.py` imports `config`
(`NetworkConfig`, `PhysicsConfig`), `attention.PhysicsInformedAttention`, and
`decoders.PortHamiltonianDecoder`. **Imported by:** `losses` (critic),
`controllers` (critic), `training`, `inference`, and the examples.

---

## 6. `losses/` — Training objective components

**Purpose** (from `__init__.py`): "Holds the composite training objective and
its components: physics, temporal, stability, IRL and HJB losses, aggregated by
`TotalLoss`." Phase 3.

| File | Key classes | Responsibility |
|---|---|---|
| `physics.py` | `PhysicsLoss` | Energy-conservation + PDE/BC/symmetry residual loss (`λ₁L_energy+λ₂L_PDE+λ₃L_BC+λ₄L_sym`). |
| `temporal.py` | `MultiStepPredictionLoss`, `TemporalSmoothnessLoss`, `AttentionRegularizationLoss`, `TemporalLoss` | Multi-step prediction error, trajectory smoothness, attention negative-entropy regularizer, and the aggregate `TemporalLoss` (§6.2). |
| `stability.py` | `LyapunovConstraintLoss`, `ParameterBoundednessLoss`, `ControlEffortLoss`, `MRASStabilityLoss` | Lyapunov-decrease penalty `mean(ReLU(V̇+margin))`, parameter L2 boundedness, quadratic control effort `E[uᵀRu]`, aggregated by `MRASStabilityLoss`. |
| `irl.py` | `IRLBellmanAccumulator`, `IRLBellmanLoss` | Trapezoidal accumulator of `∫(eᵀQe+uᵀRu)dτ` and the IRL Bellman residual loss `L_IRL = ½·E[δ_IRL²]` (§3.2, Identity 1 — model-free). |
| `hjb.py` | `HJBResidualLoss`, `LyapunovDecreaseEnforcer` | Hamilton-Jacobi-Bellman residual loss (§3.5, Identity 8) and a tighter `mean(ReLU(∇V̂·f+ℓ+margin))` decrease enforcer. |
| `adaptive_weighting.py` | `ReLoBRaLo`, `causal_weights` | Cheap, opt-in loss-balancing utilities operating on loss *values* (no extra backward pass): relative loss balancing with random lookback, and causal (temporal-ordering) PINN weights. |
| `__init__.py` | `TotalLoss` + re-exports | Weighted sum of pre-computed per-component scalar losses. |

**`TotalLoss`** takes a dict mapping component name (subset of `physics,
temporal, stability, irl, hjb, data, pcml`) to a scalar loss tensor,
weights each by the matching `LossConfig` attribute, and returns
`{"loss": total, "loss/physics": …, …}`; missing components are treated as zero.

**`adaptive_weighting.py` detail:** `ReLoBRaLo` (Bischof & Kraus,
arXiv:2110.09813) is a stateful balancer whose `weights(losses, generator=None)`
returns per-term weights (each vector summing to `num_losses`) by EMA-combining
the temperature-scaled softmax of relative-progress ratios with the running
historical weights, selecting the lookback reference (previous vs initial step)
via a reproducible Bernoulli draw (supplied generator, else a deterministic
counter); the first call stores the initial losses and returns all-ones.
`causal_weights(residuals, eps=1.0)` (Wang, Sankaran & Perdikaris,
arXiv:2203.07404) returns `w_i=exp(−eps·Σ_{k<i} residual_k)` (detached), so
`w_0=1` and later weights stay suppressed until the earlier residuals shrink —
enforcing temporal-order learning.

**Dependencies:** `physics.py`, `temporal.py`, `stability.py` have no internal
deps. `hjb.py` and `irl.py` import `models.critic.QuadraticCritic`.
`__init__.py` imports `config.LossConfig` plus the five submodules.
**Imported by:** `training/cotrain.py` (`HJBResidualLoss`, `IRLBellmanLoss`).

---

## 7. `controllers/` — Reference model, safety filter, MRAS actor

**Purpose** (from `__init__.py`): "reference models, CLF-CBF safety filter, MRAS
actor." Phase 4. The package init is docstring-only (no re-exports); the package
root imports the three classes directly from their files.

| File | Key classes | Responsibility |
|---|---|---|
| `reference_models.py` | `LinearReferenceModel` | Hurwitz linear reference model `ẋ_m = A_m x_m + B_m r`. On construction it asserts `A_m` Hurwitz, solves the Lyapunov equation for `P` (policy evaluation), and runs Kleinman iteration for `(P_opt, K_opt)`; `step()` does Euler integration. |
| `safety.py` | `CLFCBFSafetyFilter` | Closed-form CLF-CBF safety filter (Identity 3): the same `P` serves the CLF (`V=eᵀPe`) and the CBF (`h=c−eᵀPe`); projects nominal control onto the safe half-space without a QP solver. |
| `mras.py` | `MRASController` | Adaptive controller fusing classical MRAS with the actor-critic upgrade (Identities 1–4). |
| `koopman_control.py` | `KoopmanLQRController` | LQR controller on Koopman-lifted coordinates: solves the Riccati problem on a frozen model's learned latent dynamics `(A_z, B_z)` and closes the loop on the lifted tracking error. |
| `__init__.py` | (none) | Docstring-only package init. |

**`MRASController`** control law: `u(t) = u_fb(e) + K_ff·r(t) +
compensator(x_plant)`, where `u_fb = −½R⁻¹Bᵀ∇V̂ = −R⁻¹BᵀP̂e` is the
**costate-head optimal control** (Identity 2). The critic is warm-started to the
LQR/CARE solution `P_opt` at construction (so `u_fb` equals `−K_opt e` at init
and adapts thereafter via the learned `P̂`, Identity 4 fusion). The CLF-CBF
filter (Identity 3) wraps the nominal control. The DPG actor-update half is
provided by `mras_regressor` (`φ_c=[e,r,x_p]`) and `dpg_actor_step`.

**`KoopmanLQRController`** builds the latent state-cost `Q_z` (with
`include_state=True` the supplied state-cost `Q` is embedded into the leading
state block and zero elsewhere; an explicit `q_latent` penalizes the full lifted
state), reads the frozen model's `latent_matrices()`, and solves
`P_z, K_z = solve_care(A_z, B_z, Q_z, R)` at construction (registering `Q_z`,
`P_z`, `K_z` as buffers). `control(x, x_ref)` returns
`u = −(encode(x)−encode(x_ref)) @ K_zᵀ` — control on the lifted tracking error —
and `latent_gain()` exposes `K_z`. It is purely additive: it never mutates the
model or the analytic core.

**Dependencies:** `reference_models.py` imports
`utils.lyapunov` (`check_hurwitz`, `kleinman_iteration`, `solve_lyapunov`);
`safety.py` has no internal deps; `mras.py` imports
`reference_models.LinearReferenceModel`, `safety.CLFCBFSafetyFilter`,
`models.critic` (`CostateHead`, `QuadraticCritic`), and
`utils.lyapunov.solve_care`; `koopman_control.py` imports
`models.koopman.KoopmanLiftingModel` and `utils.lyapunov.solve_care`.
**Imported by:** `inference/realtime.py`,
`training/cotrain.py` (type-only), the examples, and the package root.

---

## 8. `training/` — Optimization pipelines

**Purpose** (from `__init__.py`): "physics pretrain, IRL co-train, offline IRL
trainer." Phase 5. Re-exports the three entry points.

| File | Key functions | Responsibility |
|---|---|---|
| `pretrain.py` | `pretrain_pitnn`, `data_weight_schedule`, `temporal_weight_schedule` | Three-stage physics-informed pre-training curriculum (Algorithm 2): Stage 1A physics-only, Stage 1B cosine-anneal the data weight 0.1→1.0, Stage 1C add temporal loss with linear warm-up. A validation guard halves the data weight when the physics residual spikes. |
| `cotrain.py` | `cotraining_loop` | The closed-loop actor-critic training loop ("the most critical training file", Algorithm 3 extended). Per step: PITNN forward → tracking error + MRAS control → PITNN objective (physics + optional PCML + CBF) on `optimizer_pitnn` (Adam lr=1e-4); then the critic-only updates on a separate `critic_optimizer` (Adam lr=1e-3): an opt-in HJB residual (`lambda_hjb>0`), a guarded positivity regularizer, and the IRL Bellman policy-evaluation step (grad-clip 1.0) + policy-improvement read `K=R⁻¹BᵀP̂`. |
| `irl_trainer.py` | `train_irl_critic` | Offline batch least-squares critic pre-training (§8.3): recovers `P` from the integral-RL Bellman identity `V(e(t))−V(e(t+T)) = ∫r ds` without knowing the drift; iterates and stops when `‖P̂−P_opt‖_F/‖P_opt‖_F < tol`. |
| `hinf_minmax.py` | `hji_residual`, `hinf_minmax_train`, `hinf_minmax_from_dynamics` | Neural H∞ adversarial min-max loop: three-network ADP (critic + costate protagonist + neural adversary) solving the HJI game against the analytic GARE oracle, plus the residual and a linearize-then-train bridge for dynamics callables. |
| `sac.py` | `SACTrainer` | Soft Actor-Critic learner with automatic entropy temperature: owns the policy/twin-critic/target/optimizers and one `update` of the three SAC losses + soft target update. |
| `tdmpc.py` | `tdmpc_update` | One joint TD-MPC2 world-model gradient step (latent-consistency + reward-prediction + TD value losses) over a transition batch. |
| `__init__.py` | re-exports three functions | Subpackage public surface. |

> The source design docs pin the schedules and the additions to Algorithm 3 but
> leave the base loop bodies / signatures prose-only; the function signatures,
> synthetic-trajectory generators, and returned metrics are designed in-repo and
> run on synthetic data (no external dataset — Gap G7).

**`hinf_minmax.py` detail:** `hji_residual(critic, costate, adversary, e, A, B,
D, Q, R, gamma)` returns the per-state HJI/game-Bellman residual `ρ(e) = eᵀQe +
uᵀRu − γ²‖w‖² + ∇V̂·(Ae+Bu+Dw)` with `u` the costate head and `w` the adversary;
it is differentiable w.r.t. both the critic and the adversary. `hinf_minmax_train`
co-trains the three networks two-timescale (critic minimizes `E[ρ²]` + a
positivity penalty; the adversary ascends the value; the protagonist is the
implicit slow player read off the critic), computing the analytic GARE
`(P*,K*,L*)` up front as the oracle and returning a metrics dict (per-iter
`residual`/`value`/`P_dist`/`K_dist`/`adv_dist`, the oracle/learned matrices, and
the trained modules). `hinf_minmax_from_dynamics(dynamics_fn, x0, u0, …)`
linearizes any continuous-time `f(x,u)` via `linearize_dynamics` and drives the
unchanged `hinf_minmax_train` with the extracted `(A, B)` (also returned).

**`SACTrainer`** holds a `GaussianPolicy`, a `TwinQCritic` + frozen target copy,
a learnable `log_alpha` (temperature, `alpha` property) tuned toward
`target_entropy=−action_dim`, and three Adam optimizers. `update(batch)` runs the
entropy-regularized clipped double-Q critic loss, the reparameterized actor loss,
the temperature loss, and a Polyak soft target update, returning a finite-scalar
metrics dict.

**`tdmpc_update(model, batch, optimizer, …)`** runs one joint gradient step on a
`WorldModel` over a `(s,a,r,s',done)` batch: latent-consistency
`‖next(encode(s),a) − sg[encode(s')]‖²`, reward-prediction MSE, and TD value MSE
`Q(z,a)` vs `r + γ(1−done)·value(sg[encode(s')])`; returns the per-term and total
scalar losses.

**Dependencies:** `cotrain.py` imports `losses.hjb.HJBResidualLoss`,
`losses.irl.IRLBellmanLoss` (runtime) plus `config.PITSMRASConfig`,
`controllers.mras.MRASController`, `controllers.reference_models`,
`models.PITNN`, `models.pcml.PCMLModule` (all `TYPE_CHECKING`-only).
`pretrain.py` and `irl_trainer.py` import `config`/`models`/`controllers`
symbols as type-only hints; `pretrain.py` also imports `data.TrajectoryDataset`
(type-only) and `data.make_dataloader` (lazily, only when a dataset is passed).
`hinf_minmax.py` imports `models.adversary.NeuralAdversary`,
`models.critic` (`CostateHead`, `QuadraticCritic`), `utils.linearization`, and
`utils.lyapunov.solve_gare`; `sac.py` imports `models.sac` (`GaussianPolicy`,
`TwinQCritic`); `tdmpc.py` imports `models.tdmpc.WorldModel`. **Imported by:** the
package root re-exports `pretrain_pitnn` and `cotraining_loop`.

---

## 9. `data/` — Trajectory dataset, generator, and loader

**Purpose** (from `__init__.py`): "reusable trajectory dataset, generator, and
loader." Factors the previously-inline synthetic-trajectory plumbing out of the
training pipelines into a small, **additive and opt-in** surface — importing it
has no effect on the existing training path unless a caller explicitly threads a
dataset / loader through `training` (e.g. `pretrain_pitnn(..., dataset=...)`).

| File | Key exports | Responsibility |
|---|---|---|
| `trajectory.py` | `TrajectoryDataset`, `generate_synthetic_trajectories`, `make_dataloader` | Windowed `(state, control)` dataset for the PITNN, a seedable forward-Euler synthetic generator, and a `DataLoader` convenience wrapper. |
| `__init__.py` | re-exports the three symbols | Subpackage public surface. |

**Class / function detail:**
- `generate_synthetic_trajectories(A_m, B_m, dt, n_trajectories, n_steps, control_dim, …)` — rolls out the *same* linear plant the inline co-training code uses (`ẋ = A_m x + B_m u`, forward Euler), with controls drawn i.i.d. `N(0, control_scale²)` and the initial state `N(0, init_scale²)` from a single seeded generator; returns `(states, controls)` of shape `[n_trajectories, n_steps, *]` (float32).
- `TrajectoryDataset(states, controls, *, memory_horizon)` — a `torch.utils.data.Dataset` holding one or more trajectories (single `[T, dim]`, batched `[n_traj, T, dim]`, or a list of `[T, dim]`) and yielding the *windowed* samples the PITNN consumes: for each valid current index `i ∈ [W, T−2]` it returns a dict `{state_hist [W, state_dim], control_hist [W, control_dim], state [state_dim], control [control_dim], next_state [state_dim]}`. Requires `T ≥ memory_horizon + 2`.
- `make_dataloader(dataset, batch_size, *, shuffle=True, drop_last=False, generator=None)` — a thin `DataLoader` wrapper using the default collate, which stacks the per-sample dict fields into the leading-`batch` layout the PITNN expects (`[batch, W, dim]` / `[batch, dim]`).

**Dependencies:** `trajectory.py` has **no internal dependencies** (pure
`torch` + `torch.utils.data`). **Imported by:** `training/pretrain.py`
(`TrajectoryDataset` as a type-only hint; `make_dataloader` lazily when a dataset
is passed) — the opt-in trajectory dataset/loader used by `pretrain_pitnn`.

---

## 10. `inference/` — Real-time closed-loop runtime

**Purpose** (from `__init__.py`): "real-time engine and parallel thread
architecture." Phase 6. Docstring-only package init.

| File | Key classes | Responsibility |
|---|---|---|
| `realtime.py` | `RealtimeInferenceEngine` | Seven-step closed loop: measure `x_p` → lazily init reference state + bounded `deque` buffers → PITNN forward → reference-model step + error `e=x_p−x_m` → controller forward → CBF safety filter (replaces the heuristic `V̇<0` check) → update history, return monitoring dict `{u_safe, e, v_hat, h_cbf, f_hat, cbf_active}`. `step()` is `@torch.no_grad()` and guarded by a `threading.Lock`. |
| `parallel.py` | `ControllerState`, `ParallelInferenceEngine` | Three-thread deployment scaffold: `ControlThread` (~1 kHz, calls `engine.step()`, feeds an `(e, u)` window, never blocks on adaptation), `AdaptationThread` (~100 Hz, a **real** one-step IRL Bellman critic update on a `copy.deepcopy`, then an atomic double-buffer swap of both the critic and the costate head), `MonitorThread` (~10 Hz, snapshots the CBF-activation rate). Each thread is guarded — the first exception is captured (`error`/`check()`) and fail-fast-stops the engine. Shutdown via a single `threading.Event`. |
| `__init__.py` | (none) | Docstring-only package init. |

`ControllerState` is the lock-protected snapshot dataclass (`u_safe`, `e_norm`,
`v_hat`, `h_cbf`, `cbf_active`). The docstrings flag that the engine reconciles
the real Phase-4 controller signature `MRASController.forward(e, r, x_plant,
apply_safety=True)` with the §9 spec text (it computes `V̂` itself via
`controller.critic`). The `parallel.py` adaptation step is a **real**
double-buffered IRL critic update; remaining scaffold: fixed `x_p`/`r` (no live
sensor), a cooperative `Event.wait` scheduler (not hard-real-time), and the CBF
`P` fixed at `setup_safety_filter` time.

**Dependencies:** `realtime.py` imports `controllers.mras.MRASController`,
`controllers.reference_models.LinearReferenceModel`, `models.pitnn.PITNN`, and
`models.pcml.PCMLModule` (type-only). `parallel.py` imports
`realtime.RealtimeInferenceEngine`. **Imported by:** the package root and all
three examples.

---

## 11. `examples/` — Runnable demos

**Purpose:** three CLI entry points (Phase 7) that drive the full PITNN → MRAS →
CBF stack end-to-end via `RealtimeInferenceEngine`. Each exposes `run` and
`main` and is import-safe (nothing heavy runs at import time).

| File | `run`/`main` | Scenario |
|---|---|---|
| `robotic_manipulator.py` | `run`, `main` | 2-DOF planar manipulator (IP §10.1), `H=½q̇ᵀM(q)q̇+V(q)`, sinusoidal joint reference. Four diagnostic panels: `‖e(t)‖`, `V̂(e(t))`, CBF activation flag, critic convergence `‖P̂−P_CARE‖_F/‖P_CARE‖_F`. Phase-6 acceptance gate (100-step run, no error). |
| `autonomous_vehicle.py` | `run`, `main` | Lateral control at 80 km/h (IP §10.2), wind-gust disturbance `Δ(t)=0.5·sin(2πt/10)`, with-CBF vs without-CBF lane-departure A/B (two engines differing only in the CBF projection). |
| `building_hvac.py` | `run`, `main` | Thermal-zone control (IP §10.3) with a thermal-energy Hamiltonian, energy proxy `Σu²dt` vs a simple proportional baseline, seasonal `P̂` adaptation. |

Each example imports `config` (`NetworkConfig`, `PhysicsConfig`,
`PITSMRASConfig`), `controllers.mras.MRASController`,
`controllers.reference_models.LinearReferenceModel`,
`inference.realtime.RealtimeInferenceEngine`, and `models.PITNN`.

> **Grounded caveat (from the example docstrings):** in all three demos the
> "plant" is the linear reference-model surrogate driven by
> `RealtimeInferenceEngine`; full nonlinear rigid-body / bicycle / RC-building
> dynamics are **not** simulated. They are faithful closed-loop demos of the
> PITNN → MRAS → CBF stack, not research-grade plant simulators.

---

## 12. Component Interaction — PITNN → PCML → MRASController (+ CBF) at Runtime

The runtime composition (grounded in `inference/realtime.py`,
`controllers/mras.py`, `models/pitnn.py`, and `models/pcml.py`) chains the
components as follows, once per control step:

1. **History → PITNN.** The engine pushes the measured plant state and applied
   control into bounded `deque` buffers (`maxlen = memory_horizon`), then calls
   `PITNN.forward`. Inside PITNN: normalize/embed → causal (forward-only) LSTM →
   `PhysicsInformedAttention` (context `c_t` + weights `alpha`) →
   `PortHamiltonianDecoder` producing the dynamics prediction `f̂_θ`
   (`f̂ = J∇H − [0;R_θ∂H/∂p] + B u + W_corr c_t`). `J=−Jᵀ`, `R_θ⪰0`, `H_θ>0` hold
   by construction.

2. **PCML constraint enforcement (optional).** When a `PCMLModule` wraps the
   backbone prediction, it returns the constrained prediction plus the PCML
   loss. In **soft** mode it adds `SoftPCMLLoss` residuals against the active
   `PhysicsConstraints` (`MechanicalDAE`/`HeatConductionDAE`); once the data loss
   falls below `eta` it flips to **hard** mode, projecting the prediction onto
   the DAE manifold via the `KKTProjectionLayer` (warm-started by
   `LagrangianMultiplierHead`'s `lambda_hat` derived from the same attention
   context `c_t`).

3. **Reference model → tracking error.** `LinearReferenceModel.step()` advances
   `x_m ← A_m x_m + B_m r`; the engine forms `e = x_p − x_m`. The model's `P`
   (from `solve_lyapunov`) and `P_opt/K_opt` (from `kleinman_iteration`) were
   computed at construction.

4. **MRASController.** On the tracking error `e`, the controller evaluates its
   `QuadraticCritic` for `V̂(e)` (Identity 1) and its `CostateHead` for the
   feedback `u_fb = −½R⁻¹Bᵀ∇V̂ = −R⁻¹BᵀP̂e` (Identity 2 — the action head IS the
   critic gradient). It adds the feedforward `K_ff r` and the auxiliary
   `compensator(x_plant)` to form the nominal control `u_nom`.

5. **CLF-CBF safety filter.** `CLFCBFSafetyFilter` takes `u_nom` and, using the
   **same `P`** as both CLF and CBF (`h(e)=c−eᵀPe`), closed-form projects it onto
   the safe half-space (Identity 3): `u_safe = u_nom` if the CBF margin is
   satisfied, else `u_nom` is corrected along `L_g h`. This replaces the legacy
   heuristic `V̇<0` emergency check.

6. **Apply + log + recurse.** `u_safe` is applied to the plant; the engine logs
   `{u_safe, e, v_hat, h_cbf, f_hat, cbf_active}` and pushes the new
   `(x_p, u_safe)` into the history buffers for the next step. Under
   `ParallelInferenceEngine`, step 4's critic is updated asynchronously by the
   `AdaptationThread` (deepcopy → IRL update → atomic swap) so the 1 kHz
   `ControlThread` never reads a half-updated critic.

This is exactly the layered descent of §1 run forward in time: the **utils**
math (`solve_care`, `quadratic_basis`, `make_*`) is baked into the **models**
(critic, decoder) and **controllers** (reference `P`/`K_opt`); the **models**
produce predictions and value/costate; the **controllers** + **constraints**
turn those into a safe control; and the **inference** layer closes the loop.
