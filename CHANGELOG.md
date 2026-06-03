# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-06-03

A behavior- and API-preserving **minimize / simplify / optimize** pass (design:
`docs/superpowers/specs/2026-06-03-v0.3.1-simplification-design.md`), bundled with
the architecture-tooling, graph-backed docs, and deferred-item resolution that had
accumulated since v0.3.0. No public-API or `PITSMRASConfig` changes; the full test
suite stays green throughout.

### Simplified / optimized (v0.3.1 pass)

- **Quadratic-basis consolidated (DRY).** The upper-triangular `eᵢeⱼ` basis
  convention (`diag = P[i,i]`, off-diag `= P[i,j] + P[j,i]`) was independently
  re-encoded in three files. `utils/lyapunov.py` is now the single source of
  truth via new vectorized `pack_symmetric`/`unpack_symmetric` helpers (and a
  last-axis-indexing `quadratic_basis`); `QuadraticCritic.extract_P`/`set_P` and
  the IRL trainer delegate to them (three hand-rolled loops removed).
  Behavior-identical (new equivalence tests; existing round-trip/convergence
  tests stay green).
- **PITNN `forward` dict slimmed.** Dropped the redundant `f`/`H` keys (exact
  aliases of `f_hat`/`H_val`, consumed only by one test); the returned dict is
  now `f_hat`/`H_val`/`context`/`alpha`/`h_enc`/`P_diss`/`energy_loss`/
  `attn_reg_loss` (+ `lam_hat` when a Lagrangian head is attached).
- **KKT projection perf.** When the Newton loop converges via its tolerance
  break, the implicit-function one-step reuses the already-computed constraints
  + Jacobian instead of recomputing them (output-identical; ~18.1 → 16.4 ms per
  projection call).

### Added

- **`tools/` developer utilities** (ported/copied from the nanoclaw repo):
  `chunking-for-files` and `compress-for-context` (generic TS text utilities,
  as-is) plus **`create-dependency-graph`, rewritten as a standalone Python
  tool** (`tools/create-dependency-graph/create_dependency_graph.py`). It parses
  Python imports (relative + absolute intra-package, incl. parenthesized
  multi-line), `__all__` + public top-level defs/classes/constants as exports,
  `__init__.py` barrel re-exports, `TYPE_CHECKING`-guarded imports (the type-only
  analog), real import cycles, unused files/exports, and test coverage — emitting
  the report set under `docs/architecture/`. Runs on the repo's own Python (no
  Node toolchain); 10 unit tests in `tools/create-dependency-graph/`.
- **`docs/architecture/` graph-backed documentation.** Generated tool outputs
  (`DEPENDENCY_GRAPH.md`, `dependency-graph.{json,yaml}`,
  `dependency-summary.compact.json`, `TEST_COVERAGE.md`, `test-coverage.json`,
  `unused-analysis.md`) plus five hand-written, graph-grounded docs:
  `OVERVIEW.md`, `COMPONENTS.md`, `API.md`, `DATAFLOW.md`, and a refreshed
  `ARCHITECTURE.md`. The graph reports 39 first-party files across 10 modules,
  ~5,219 LOC, 114 exports (45 re-exported), **0 circular dependencies, 0 unused
  files/exports**.
- **`examples/pcml_heat_diffusion.py`** — coordinate-bearing hard-PCML demo: a
  small MLP `T(x,t)` trained on the 1-D heat equation with genuine `(x, t, ∂)`
  autodiff derivatives; soft PCML reduces the residual and the KKT projection
  drives the point-wise violation to ~0.
- **`train_irl_critic_gd`** (`training/irl_trainer.py`) — an offline
  gradient-descent IRL critic fit on fixed optimal-closed-loop data (decoupled
  from control-loop stability; converges reliably from an arbitrary `P`),
  returning the convergence history.
- **`critic_convergence`** metric in `cotraining_loop` (per-step
  `‖P̂ − P_opt‖_F / ‖P_opt‖_F`).

### Fixed

- **`MechanicalDAE` `ConstraintSpec` widths** — the spec counted "equation
  groups" (`n_differential = 1`) while the EOM residual is `n_joints`-wide, which
  would malform the KKT projection on mechanical systems. The spec now reports
  the true residual vector widths (`n_differential = n_joints + 2·n_holonomic`;
  `n_inequality = 2·n_joints`; `n_outputs = 2·n_joints`), and the KKT projection
  on a holonomic `MechanicalDAE` is verified (violation < 1e-3).

### Changed

- Moved `docs/ARCHITECTURE.md` → `docs/architecture/ARCHITECTURE.md` and added a
  §0 "Implemented Architecture (v0.3.0)" graph-backed as-built summary; README
  documentation section now links the `docs/architecture/` set.
- **Example fidelity:** the robotic-manipulator demo now genuinely trains the
  critic (panel (d) is a real IRL convergence curve), and the autonomous-vehicle
  demo's CBF comparison is non-vacuous (lane-hold under a strong gust with a tight
  ellipsoid; the filter engages ~11% and bounds the departure / safe-set
  violation), with an added CBF-activation panel and honest framing.

## [0.3.0] - 2026-06-02

The **PCML** (Physics-Constrained Machine Learning) component — soft + hard
physics-constraint enforcement (Patel et al. 2022 and DAE-HardNet,
arXiv:2512.05881) — plus the pre-PCML audit remediation that makes v0.2.0
faithful to the Implementation Plan §3 identities before the new layer is built
on top.

### Added

- **PCML constraints library** (`src/pits_mras/constraints/`, PCML Addendum §2.1):
  the `PhysicsConstraints` ABC + `ConstraintSpec`, plus `MechanicalDAE`
  (Euler-Lagrange equations of motion with optional holonomic constraints) and
  `HeatConductionDAE` (1-D transient heat conduction). Each exposes the
  differential / equality / inequality residuals that feed both the soft PCML
  loss and the hard KKT projection, with a shared `violation` metric.
- **PCML soft path + supporting layers** (`src/pits_mras/models/pcml.py`,
  `src/pits_mras/models/lagrangian_head.py`): `SoftPCMLLoss` (Patel et al. 2022
  augmented loss `λ_diff‖D‖² + λ_eq‖h‖² + λ_ineq‖ReLU(g)‖²`),
  `TaylorNeighborhoodApproximation` (DAE-HardNet §3 multi-point neighborhood
  that turns differential operators into algebraic variables), and
  `LagrangianMultiplierHead` (KKT warm-start multipliers; inequality duals ≥ 0).
- **PCML hard path** (`src/pits_mras/models/pcml.py`): `KKTProjectionLayer` —
  a differentiable projection onto the DAE constraint manifold
  (`min ½‖y−ŷ‖² s.t. D=0, h=0, g≤0`) via a Newton solve on the KKT system with
  Fischer-Burmeister complementarity; gradients flow through a single
  implicit-function-theorem step (no unrolling). `PCMLModule` unifies the soft
  and hard modes with DAE-HardNet dynamic activation at the `eta` threshold
  (loss `MSE(ỹ,ȳ) + ω·MSE(d̃, AD(∂ỹ))` in hard mode). Verified against the
  closed-form linear-equality projection and a heat-equation violation drop to
  `< 1e-4`.
- **PCML pipeline integration (opt-in, backward-compatible)**: `PCMLConfig` on
  the master `PITSMRASConfig` (soft/hard params + constraint selection);
  `LossConfig.lambda_pcml` and a `pcml` component in `TotalLoss`; an optional
  `lagrangian_head` on `PITNN` that, when supplied, emits `lam_hat` (KKT
  warm-start multipliers) without changing the default v0.2.0 output contract.
- **PCML loop hooks (opt-in)**: `cotraining_loop` accepts a `pcml_module` that
  adds the constraint loss (soft, escalating to the hard KKT projection at the
  `eta` data-loss threshold — DAE-HardNet §3.1 dynamic activation) and records a
  `pcml_loss` metric; `RealtimeInferenceEngine` accepts a `pcml_module` +
  `pcml_projection_tolerance` to project `f_hat` onto the manifold at inference,
  bypassing the projection when the violation is already below tolerance
  (DAE-HardNet §4.8). Both default to `None` (v0.2.0 loops unchanged).

### Fixed

- **Audit remediation — mathematical faithfulness to the §3 identities** (a
  four-part pre-PCML pass; baseline was 139/139 green, now 146/146):
  - **Costate optimal control (Identity 2):** `CostateHead.u_opt` was missing the
    ½ factor and returned `-2Ke` instead of the LQR gain `-Ke`. Added a
    `half_grad=True` parameter mirroring `HJBResidualLoss`; `lambda_hat` remains
    the true costate `∇V̂ = 2P̂e` (`models/critic.py`).
  - **Port-Hamiltonian dissipation (§3.1):** the soft energy loss was internally
    inconsistent (`f_diss` used the finite-difference `q̇` while `P_diss` used
    `∇_qH`), so the passivity residual could not vanish. Dissipation now acts on
    the momentum block via the Hamiltonian velocity `∂H/∂p`, with
    `P_diss = (∂H/∂p)ᵀ R (∂H/∂p)`, so `‖dH/dt − P_control + P_diss‖²` vanishes by
    construction (`models/decoders.py`). This is exactly the soft-constraint
    weakness the forthcoming PCML hard-projection layer supersedes.
  - **MRAS feedback wiring (Identities 2 & 4):** feedback now routes through the
    costate head (`u_fb = -R⁻¹BᵀP̂e`) so the learned critic drives control; the
    critic is warm-started to `P_opt` in `__init__`, so `u_fb = -K_opt e` at
    initialization and adapts thereafter (previously a frozen `K_fb = K_opt`
    buffer was used and the critic never reached the actuator). `forward` now
    also returns `lambda_hat`/`v_hat`; `inference/realtime.py` wraps the
    controller call in `enable_grad` and detaches its output
    (`controllers/mras.py`, `inference/realtime.py`).
  - **DPG actor (Identity 4):** added `mras_regressor` (`φ_c = [e, r, x_p]`),
    `dpg_action_value_gradient` (`Ru + BᵀP̂e`, which vanishes at the optimal
    control), and `dpg_actor_step` (the deterministic policy gradient on the
    actor parameters `K_ff`/`compensator`, with the critic left to IRL)
    (`controllers/mras.py`).

### Changed

- Warm-starting the MRAS critic to `P_opt` changes the CBF `P` built by
  `setup_safety_filter` (via `extract_P`) and makes the `robotic_manipulator`
  critic-convergence panel start near zero — the mathematically-correct intended
  behavior, not a regression.

### Documentation

- Added the source design docs `docs/PITS-MRAS — Implementation Plan.md` and
  `docs/PITS-MRAS — PCML Addendum.md` for in-repo reference.

### Note

- Commits `034082a` and `3222c42` added CHANGELOG entries for example fixes
  (autonomous_vehicle CBF, robotic_manipulator critic training) whose code edits
  failed to save during a tool-channel outage — the entries described work the
  source never received. Those unbacked entries have been removed here; the
  example improvements remain genuine TODOs (see the v0.2.0 known-limitations note).

## [0.2.0] - 2026-05-31

First release with all nine ROADMAP phases implemented (the 0.1.0 baseline was a
scaffold). Highlights below.

### Added

- **Phase 9 — CI/CD finalization** (`docs/ROADMAP.md` Phase 9 / §12): the project
  is now fully built out — all 9 ROADMAP phases complete.
  - `.github/workflows/ci.yml` — finalized: the lint step now covers `examples`
    (`flake8 src tests examples`) and the test step enforces the coverage gate in CI
    (`pytest --cov=pits_mras --cov-fail-under=60`). Matrix Python 3.10/3.11/3.12,
    CPU-only torch, on push + pull_request.
  - `.pre-commit-config.yaml` — new: black / isort / flake8 / mypy hooks, sharing
    config with `pyproject.toml` + `setup.cfg` so hooks match CI exactly.
  - Verified locally: `flake8 src tests examples` → 0, `mypy src` → 0, the CI test
    command reports "Required test coverage of 60% reached. Total coverage: 98.32%",
    139 passed.

- **Phase 8 — Tests / coverage** (`docs/ROADMAP.md` Phase 8 / §11, §13): the full
  test suite now runs with **no skips**, and `pytest --cov=pits_mras` reports **98%**
  (above the §13 ≥60% gate).
  - Implemented the last placeholder, `test_irl_critic_converges_to_lyapunov_P`
    (Identity 1, fully realized): `train_irl_critic` fits the critic from model-free
    trajectory data and the recovered P̂ matches the CARE `P_opt` to rel-err 0.4 %
    (< the trainer's 1 % stop tolerance); the fitted critic's `extract_P()` matches too.
  - Added targeted coverage tests for the lowest-covered modules: parallel-inference
    engine start/stop + state updates, the IRL-trainer non-convergence path, and the
    pre-training spike-safeguard + curriculum stage boundaries (`training/cotrain.py`
    54→81 %, `training/pretrain.py` 58→82 %).

- **Phase 7 — Examples** (`docs/ROADMAP.md` Phase 7): runnable closed-loop demos
  replacing the example stubs:
  - `examples/robotic_manipulator.py` — 2-DOF arm, sinusoidal joint reference,
    4-panel diagnostic figure (‖e‖, v̂, CBF-activation flag, critic-convergence
    ‖P̂−P_CARE‖/‖P_CARE‖). The §13 acceptance gate (100-step headless run generates
    a figure, exit 0) passes.
  - `examples/autonomous_vehicle.py` — lateral control with wind-gust
    Δ(t)=0.5·sin(2πt/10); with-CBF vs without-CBF comparison.
  - `examples/building_hvac.py` — thermal-zone control vs a proportional baseline.
  - Each exposes `run(steps, show=False) -> dict` and is import-safe (all sim/plot
    work under `run()`/`main()`; matplotlib forced to Agg); new `tests/test_examples.py`
    (10) runs them headless. Independently reviewed (APPROVE).
  - **Simplifications (flagged in-code):** toy linear surrogate dynamics, and the
    examples run *inference only* on an untrained critic (no training loop), so the
    critic-convergence panel is a diagnostic path, not a convergence claim.

- **Phase 6 — Inference Engine** (`docs/ROADMAP.md` Phase 6): real torch
  implementations replacing the inference stubs:
  - `inference/realtime.py` — `RealtimeInferenceEngine`: thread-safe `@torch.no_grad`
    closed-loop `step()` (§9.1) — bounded `deque(maxlen=horizon)` history → PITNN
    forward → reference-model step → tracking error e=x_p−x_m → controller → CBF
    control; returns `{u_safe, e, v_hat, h_cbf, f_hat, cbf_active}`. The PITNN forward
    is wrapped in `enable_grad` (its decoder needs autograd for ∇H) with the output
    detached, so the no_grad/Lock contract holds and no graph leaks.
  - `inference/parallel.py` — `ParallelInferenceEngine` reference skeleton +
    `ControllerState` dataclass (§9.2): ControlThread / AdaptationThread /
    MonitorThread with a `threading.Event` shutdown and a deepcopy→swap critic
    double-buffer.
  - Adapted to the real Phase 1–5 APIs (the §9 spec text predated them): controller
    called as `forward(e, r, x_plant)`; CBF control read from key `u`, slack from
    `slack`; `v_hat` computed via `controller.critic(e)`.
  - Tests: un-skipped `test_full_forward_pass_no_crash` (`tests/test_smoke.py`); new
    `tests/test_inference.py`. Two independent reviews (APPROVE_WITH_NITS, 0 blocking):
    closed-loop semantics verified (e=x_p−x_m, deques bounded, finite over 12+ steps);
    parallel engine starts/ticks/stops with no deadlock.

- **Phase 5 — Training Pipelines** (`docs/ROADMAP.md` Phase 5): real torch
  implementations replacing the training stubs:
  - `training/pretrain.py` — `pretrain_pitnn`: three-stage curriculum (§8.1) —
    1A physics-only, 1B cosine-annealed data weight 0.1→1.0, 1C linear temporal
    warm-up — with the spike-detection safeguard (halve λ_data + warn). Collocation
    uses smooth trajectories so the PITNN's internal finite-difference velocity stays
    bounded.
  - `training/cotrain.py` — `cotraining_loop`: closed-loop actor-critic loop with the
    §8.2 additions — IRL critic update (separate Adam lr=1e-3, grad-clip 1.0, policy
    improvement K←R⁻¹BᵀP̂), HJB term, costate consistency, critic positivity, CBF
    constraint; PITNN Adam lr=1e-4. The IRL step runs after the PITNN step (autograd
    ordering).
  - `training/irl_trainer.py` — `train_irl_critic`: offline batch least-squares critic
    fit; stops at ‖P̂−P_opt‖_F/‖P_opt‖_F < 0.01 (§8.3).
  - `__init__.py` — enabled the previously-deferred `pretrain_pitnn`/`cotraining_loop`
    re-exports (the eight-symbol top-level API is now complete).
  - Tests: un-skipped `test_pretrain_one_epoch` and `test_cotrain_one_episode`
    (`tests/test_smoke.py`); new `tests/test_training.py` (13). Acceptance gate
    `pytest tests/test_smoke.py` passes (Phase-5 tests green; full-forward stays
    skipped for Phase 6). Independently reviewed (APPROVE_WITH_NITS): IRL trainer
    converges to rel-err 8.6e-8 (<0.01); the critic genuinely steps (Δ=0 when
    critic_lr=0); schedule boundary values exact.
  - **Note (G5/G6):** §8.2's base co-training loop is prose-only and its variable
    names don't match the implemented Phase 1–4 APIs, so the loop and the
    `plant_dt`/`excitation` knobs were designed to wire the real modules together.

- **Phase 4 — Controllers** (`docs/ROADMAP.md` Phase 4): real torch
  implementations replacing the controller stubs:
  - `controllers/reference_models.py` — `LinearReferenceModel`: Hurwitz-checked
    reference dynamics ẋ_m=A_m x_m+B_m r with forward-Euler `step` (§7.1).
  - `controllers/safety.py` — `CLFCBFSafetyFilter`: closed-form single-constraint
    CBF projection with h(e)=c−eᵀPe, L_f h=−2eᵀP A_m e, L_g h=−2eᵀPB (§3.4/§7.2);
    plus `cbf_constraint_loss` soft penalty. Forward-invariance verified (eᵀPe stays
    ≤ c over a 100-step closed-loop sim under destabilizing nominal control).
  - `controllers/mras.py` — `MRASController`: actor-critic control u=−K_fb·e+K_ff·r
    with optional CBF filtering; `lqr_warm_start(Q,R)` sets K_fb to the CARE gain and
    warm-starts the critic P (§7.3).
  - `models/critic.py` — added `set_P` (inverse of `extract_P`) so the critic can be
    warm-started to the CARE solution; round-trip verified.
  - Tests: un-skipped the three CBF gate tests (`test_cbf_projects_unsafe_control`,
    `test_cbf_identity_when_safe`, `test_cbf_forward_invariance`) and
    `test_optimal_control_equals_lqr_gain`; new `tests/test_controllers.py` (11).
    Acceptance gate `pytest tests/test_safety.py tests/test_identity_costate.py`
    passes (5/5). Independently reviewed (APPROVE): forward invariance holds
    (max eᵀPe=0.9999≤1.0), K_fb matches the CARE gain to 1.7e-7, costate-derived
    control equals −Ke to 1.4e-7.

- **Phase 3 — Loss Functions** (`docs/ROADMAP.md` Phase 3): real torch
  implementations replacing the loss stubs:
  - `losses/physics.py` — `PhysicsLoss`: port-Hamiltonian energy balance
    residual `dH/dt − (P_control − P_diss)` plus optional PDE/BC/symmetry
    residuals, λ-weighted (Implementation Plan §6.1).
  - `losses/temporal.py` — `MultiStepPredictionLoss`, `AttentionRegularizationLoss`,
    `TemporalSmoothnessLoss`, `TemporalLoss` (§6.2; attention-weighted multi-step
    prediction).
  - `losses/stability.py` — `LyapunovConstraintLoss` (penalize V̇>0),
    `ParameterBoundednessLoss`, `ControlEffortLoss` (uᵀRu), `MRASStabilityLoss`.
  - `losses/irl.py` — `IRLBellmanAccumulator` + `IRLBellmanLoss`: the Integral-RL
    Bellman residual δ_IRL = ∫(eᵀQe+uᵀRu)dτ − [V̂(t)−V̂(t−T)], L=½E[δ²] (§3.2).
    Model-free: the drift matrix A does not appear (verified numerically — δ_IRL≈0
    when V̂ is the true value function).
  - `losses/hjb.py` — `HJBResidualLoss` (§3.5; u*=−R⁻¹Bᵀ∇V̂, default λ=0.01) +
    `LyapunovDecreaseEnforcer`. Residual ≈0 at the LQR optimum (verified).
  - `losses/__init__.py` — `TotalLoss` aggregator combining the sub-losses with
    `LossConfig` weights, returning the total plus per-component logging scalars.
  - New tests `tests/test_losses.py` (17); replaced the duplicate-class stub in
    `tests/test_irl.py` with the two mandated tests (un-skipped). Acceptance gate
    `pytest tests/test_irl.py tests/test_identity_costate.py` passes. Independently
    reviewed (APPROVE_WITH_NITS): δ_IRL=2.9e-7 at true value, HJB residual=8.4e-13
    at LQR optimum, A confirmed absent from the IRL loss.

- **Phase 2 — Neural Network Models** (`docs/ROADMAP.md` Phase 2): real torch
  implementations replacing the model stubs:
  - `models/attention.py` — `PhysicsInformedAttention`: three-headed attention
    (temporal scaled-dot-product, physical, error-driven cosine) fused by a learned
    3-way softmax gate; `attention_regularization_loss`.
  - `models/decoders.py` — `HamiltonianNet` (Softplus → H>0), `DissipationNet`
    (Cholesky → R=LᵀL⪰0), `PortHamiltonianDecoder` (f̂=J∇H − R·q̇ + B·u + W_corr·c_t,
    enforcing the §3.1 identities; ∇H via autograd `create_graph=True`).
  - `models/critic.py` — `QuadraticCritic`: V̂(e)=eᵀP̂e with P̂=LᵀL+εI; `costate`
    returns λ=2P̂e (Identity 2: costate = critic gradient = ∇V̂).
  - `models/pitnn.py` — `PITNN`: LSTM encoder → physics-informed attention →
    port-Hamiltonian decoder; `forward(...)` returns `f`, `h`, `context`, `alpha`,
    `h_enc` plus monitoring keys `f_hat`, `h_val`, `p_diss`, `energy_loss`,
    `attn_reg_loss`. (The critic is standalone in `critic.py`; it is wired into the
    control loop by the Phase-4 controller, per spec §7.3 — PITNN does not embed it.)
  - Tests added to `tests/test_models.py` (attention shapes / alpha-sum, decoder
    shapes + backprop, critic value/costate/positivity, PITNN forward dict);
    un-skipped the three model gate tests (`test_dissipation_matrix_psd`,
    `test_J_skew_symmetric`, `test_hamiltonian_positive`) and, in
    `tests/test_identity_costate.py`, `test_costate_equals_grad_V`.
  - Acceptance gate (`pytest tests/test_models.py`) passes (12/12). Independently
    reviewed (APPROVE_WITH_NITS): the out-of-place decoder assembly is autograd-safe
    and numerically equivalent to the spec; `QuadraticCritic.costate` matches autograd
    ∇V̂ to 0.0.

- **Phase 1 — Foundation Layer** (`docs/ROADMAP.md` Phase 1): real implementations
  replacing the stubs in `src/pits_mras/`:
  - `config.py` — six dataclasses (`NetworkConfig`, `PhysicsConfig`, `MRASConfig`,
    `SafetyConfig`, `LossConfig`, `TrainingConfig`) + master `PITSMRASConfig` with
    `from_yaml`/`to_yaml`; field names and defaults per Implementation Plan §4.2.
  - `utils/lyapunov.py` — `solve_lyapunov`, `kleinman_iteration`, `solve_care`,
    `check_hurwitz`, `lyapunov_derivative`, `quadratic_basis` (scipy-backed; the
    P-matrix engine). Solves AᵀP+PA=−Q (transpose convention verified numerically).
  - `utils/hamiltonian.py` — `make_skew_symmetric` (J=−Jᵀ), `make_positive_definite`
    (R=LᵀL⪰0), `port_hamiltonian_energy_loss`, `hamiltonian_positivity_loss` (torch).
  - `utils/pe_monitor.py` — `PEMonitor` (Gram min-eigenvalue persistence-of-excitation
    check + probing-noise helper).
  - New tests: `test_config.py`, `test_lyapunov_utils.py`, `test_hamiltonian_utils.py`,
    `test_pe_monitor.py`; un-skipped `test_kleinman_converges_to_care` and
    `test_quadratic_basis_reconstructs_P`.
  - Acceptance gate (ROADMAP Phase 1) passes: `solve_lyapunov(-I, I) → 0.5·I`.

### Changed

- **Packaging reconciled (ROADMAP gap G2, resolved per plan):** `setup.py` distribution
  `name` → `pits_mras`, `version` → `0.1.0`, `python_requires` → `>=3.10`.
  `requirements.txt` replaced with the Phase-1 runtime set (`numpy`, `scipy`, `torch`,
  `pyyaml`); `control` intentionally omitted (G3 — scipy provides the solvers).
  `src/pits_mras/__init__.py` `__version__` → `0.1.0` and re-exports the six available
  stub classes (`pretrain_pitnn`/`cotraining_loop` deferred to Phase 5 — not yet defined).

### Added (foundation, prior)

- **`docs/ARCHITECTURE.md`** — design/architecture blueprint distilled from the two
  design PDFs in `docs/` (the *Mathematical and Architectural Blueprint* and the
  *Complete Implementation Plan*). Documents the three-paradigm merger (PINN +
  time-series deep learning + MRAS), the canonical module layout, the mapping of the
  ten RL/optimal-control identities to owning modules, the new loss terms / network
  heads / re-derived unified adaptation law, the data flow (with a mermaid diagram),
  the training & inference pipeline, and the stability/safety/testing strategy.
- **`docs/ROADMAP.md`** — phased roadmap operationalizing the Implementation Plan:
  9 build phases (Foundation → CI/CD) grouped into 4 milestones, each with
  deliverables, dependencies, acceptance gates, and checkbox task lists. Priorities
  follow the blueprint's highest-leverage-first ordering (Integral-RL policy
  evaluation + CLF-CBF-QP safety filter). Source gaps/conflicts are flagged (G0–G9),
  including the Blueprint-vs-Plan disagreement on the third network head
  (adversary/H∞ vs costate) and the `setup.py`/`requirements.txt` naming conflict.
- **`src/pits_mras/` package scaffold** — package tree matching the architecture's
  canonical layout: `models/`, `controllers/`, `losses/`, `training/`, `inference/`,
  `utils/`, plus `config.py`. Each module is a documented stub (purpose + owning
  phase + relevant identity + `TODO(phase-N)`); modules with a named API expose a
  stub class raising `NotImplementedError`. `config.py` uses stdlib `dataclasses`
  (the design's stated choice; no pydantic dependency introduced).
- **Test skeleton** — `tests/test_imports.py` smoke test importing every module in
  the package, plus the six identity/safety/model/IRL/smoke test files from the plan
  with their mandated test names as `@pytest.mark.skip` placeholders pending
  implementation. Also three `examples/` stubs.
- **CI + tooling** — GitHub Actions workflow (`.github/workflows/ci.yml`) running
  flake8, mypy, and pytest+coverage across Python 3.10–3.12; `setup.cfg` (flake8
  config); `pyproject.toml` (black/isort/mypy/pytest config); `requirements-dev.txt`
  (dev toolchain).

### Notes

- **Status:** ALL 9 phases (foundation, models, losses, controllers, training,
  inference, examples, full test suite, CI/CD) are implemented and tested. The
  framework is built out per `docs/ROADMAP.md`.
- Verified gates: `flake8 src tests examples` → 0; `mypy src` → 0;
  `pytest` → 139 passed, 0 skipped; coverage 98 % (≥60 % gate enforced in CI);
  `import pits_mras` → 0.2.0.
- **Known limitations (genuine TODOs):** the `examples/` are toy/synthetic demos
  on a linear surrogate plant. `autonomous_vehicle` runs the CBF at its default
  margin where the filter does not engage, so the with-/without-CBF comparison is
  currently not illustrative; `robotic_manipulator` runs the critic without
  training, so its convergence panel is a static diagnostic. Improving both, and
  building out the H∞ disturbance/adversary head (gap G1, Blueprint Connection 7),
  remain future work.
- **CI install:** still `pip install -e . --no-deps` plus the dev toolchain in the
  workflow. Phase 1 utils import numpy/scipy/torch, so CI now also installs the
  Phase-1 runtime deps (CPU-only torch) before running the gates.
- **Python floor:** the Implementation Plan stated a 3.9 baseline, but the current
  mypy release dropped `python_version = 3.9` support and `torch>=2.0.0` requires
  3.10+, so the CI matrix and tool configs use **3.10/3.11/3.12**.
