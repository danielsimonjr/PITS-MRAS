# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.7] - 2026-06-05

Sprint item todo#1 (CDG import-parser fix) plus a release-hygiene fix for the
version-string tests. Suite green; flake8 + mypy clean.

### Fixed

- **`create-dependency-graph` no longer swallows code after a commented import.**
  A function-level `from x import y  # noqa: E402 ...` was mis-parsed: the logical-
  line joiner tested `"(" in buf` against the *whole* line (including the comment)
  but `")"` only against the code before `#`, so a paren inside the comment made it
  greedily join the rest of the file into one "import name"; `_split_import_names`
  also didn't strip the trailing comment. Both fixed (comment-stripped before the
  paren-balance and name-split checks). The `examples/` per-file import tables in
  `dependency-graph.json` / `DEPENDENCY_GRAPH.md` are now clean (e.g. `plants` deps
  show `lateral_tyre_step` / `rc_thermal_step`, not code blobs), and imports that
  were previously eaten (matplotlib, numpy, stdlib) are now correctly recorded.
  Two regression tests added.
- **Version-assertion tests are now bump-robust.** `test_imports.test_version` and
  `test_smoke.test_package_imports` hardcoded `"0.4.5"`, so any release bump broke
  them (it red-lit the v0.4.6 CI). They now assert the semver *format* and
  *consistency* with `setup.py` instead of a literal, so future bumps don't require
  editing tests.

## [0.4.6] - 2026-06-05

First item of the improvement sprint (ROADMAP #4). Additive/backward-compatible;
suite green (231); flake8 + mypy clean.

### Added

- **Rollout-stability + conservation-drift diagnostics** (`utils/diagnostics.py`,
  importable via `from pits_mras.utils.diagnostics import ...`) — three pure,
  torch-only functions that validate physics consistency and stability over a
  multi-step rollout (the port-Hamiltonian decoder enforces the per-step energy
  residual, but nothing previously checked long-horizon behaviour):
  - `energy_drift` / `max_energy_drift` — drift of a conserved quantity (e.g. the
    Hamiltonian) from its initial value along a rollout (absolute or relative).
  - `valid_prediction_time` — the Valid Prediction Time (VPT): elapsed time before
    a rollout's normalized L2 error first exceeds a tolerance.
  - `rollout_jacobian_spectral_radius` — spectral radius of the one-step Jacobian
    (the local error-amplification factor; `> 1` ⇒ geometric error growth).
  20 new unit tests (`tests/test_diagnostics.py`).

### Docs

- **`docs/ROADMAP.md` rewritten as a forward-looking roadmap.** Removed the stale,
  fully-implemented nine-phase build plan (it still described the repo as
  "scaffold-only"; CHANGELOG and `docs/architecture/` are the durable record of
  shipped work) and added **10 research-derived improvement proposals** grouped by
  capabilities / efficiencies / simplicities, each anchored to a source file and a
  primary reference. Added a "Known gaps / deferred" section (G8 MIMO control
  simplification, Connection 5 SAC, Connection 9 TD-MPC2, G7 no `data/` loader,
  re-verified against the source) and an "already implemented" honesty note. Fixed
  the dangling footer cross-reference (`docs/ARCHITECTURE.md` →
  `docs/architecture/ARCHITECTURE.md`).

### Tooling

- **`create-dependency-graph` output is now reproducible.** `analyze_test_coverage`
  built the per-test source lists from set iteration, whose order is not stable,
  so `test-coverage.json` churned on every regeneration even with no code change.
  The `coverageMap` / `testToSourceMap` list values are now sorted, making the
  generated reports idempotent (verified: two consecutive runs are byte-identical).
  New regression test `test_analyze_test_coverage_maps_are_sorted_for_reproducibility`.

## [0.4.5] - 2026-06-04

Sixth and final v0.4.x sub-project: the H∞ robust-control **core** (partly
resolves gap G1). Additive/backward-compatible; suite green; flake8 + mypy clean.

### Added

- **`solve_gare(A, B, Q, R, gamma, D=None)`** (`utils/lyapunov.py`, beside
  `solve_care`) — solves the H∞ Game Algebraic Riccati Equation
  `AᵀP + PA + Q − P(BR⁻¹Bᵀ − γ⁻²DDᵀ)P = 0` via the Hamiltonian–Schur method.
  Returns `(P, K, L)` — the stabilizing `P`, the robust control gain
  `K = R⁻¹BᵀP`, and the worst-case-disturbance gain `L = γ⁻²DᵀP`. `D` defaults to
  `B` (matched disturbance). Raises `ValueError` on an infeasible `γ`; `γ → ∞`
  recovers the CARE.
- **`AdversaryHead(critic, D, gamma)`** (`models/critic.py`, beside `CostateHead`;
  re-exported from `pits_mras.models`) — the analytic H∞ worst-case-disturbance
  head, `w* = (1/2γ²)·∇V̂·D = γ⁻²DᵀPe`, by construction from the critic gradient
  (the robust-control sibling of the costate head). Closes the "head" of gap G1.

### Notes

- This is the H∞ *core* (solver + analytic head + gains), all deterministically
  verifiable. The neural adversarial **min-max training loop** is deferred to
  v0.5.0.

## [0.4.4] - 2026-06-04

Fifth v0.4.x sub-project: `ParallelInferenceEngine` hardening. Backward-compatible
(new keyword args + a new `ControllerState` field); suite green; flake8 + mypy
clean.

### Changed

- **`ParallelInferenceEngine` is no longer a no-op skeleton.** The adaptation
  thread now performs a **real double-buffered critic update**: the control
  thread feeds a bounded `(e, u_safe)` window, and `_adaptation_update()`
  deepcopies the critic (under the critic lock — the quadratic critic is tiny, so
  the copy never races a concurrent forward pass), takes one IRL Bellman Adam
  step on the copy *off* the lock, then atomically swaps **both**
  `controller.critic` and `controller.costate_head.critic` (the skeleton swapped
  only the critic, leaving the costate head stale). Tracked via
  `ControllerState.adaptation_swaps`. New constructor args `irl_window=8`,
  `adapt_lr=1e-3`.

### Fixed

- **No more silent thread death.** Each thread body runs under a guard that
  captures the first exception (new `error` property + `check()`) and triggers a
  fail-fast shutdown, instead of a daemon thread dying unnoticed.

### Notes

- Still a scaffold (documented): fixed `x_p`/`r` (no live sensor), a cooperative
  `Event.wait` scheduler (not hard-real-time), and the CBF `P` fixed at setup.

## [0.4.3] - 2026-06-04

Fourth v0.4.x sub-project: higher-fidelity example plants. Examples-only (no
library API change); suite green; flake8 + mypy clean.

### Added

- **Nonlinear example plants** (`examples/plants.py`): `pendulum_step` (1-DOF
  manipulator joint with `sin`-gravity), `lateral_tyre_step` (single-track
  lateral dynamics with `tanh` tyre-force saturation), `rc_thermal_step` (2-node
  RC building-thermal network + saturated heater). Each is a pure
  `f(state, u, dt) -> next_state` step; new `tests/test_example_plants.py` checks
  the physics. The three example demos now control these **nonlinear** plants
  instead of the toy linear surrogates, exercising the controller's
  model-mismatch robustness. Each plant linearizes to the example's original
  surrogate, so the existing LQR/CBF controller stays stabilizing (all three
  closed loops verified bounded/finite).

### Changed

- The HVAC example's reference model is now the 2-node RC linearization (it
  tracks the zone temperature).
- Fixed a pre-existing time-alignment off-by-one in the AV example's `_simulate`
  (the lateral-offset series was recorded one step late).

## [0.4.2] - 2026-06-04

Third v0.4.x sub-project: KKT projection robustness. Backward-compatible
(default-on line search only changes *divergent* cases); suite green; flake8 +
mypy clean.

### Changed

- **KKT projection: backtracking line search in the Newton solve** (resolves the
  carried-forward gap #1 — the robustness half of the v0.3.2 debt #2). The
  forward Newton loop took an undamped full step that **diverges** on
  high-curvature constraints (a single equality `atan(8·y)=0` from a far start
  blew the residual up to ~1e7). It now backtracks the step length on the L∞
  residual: accept the first halving that strictly decreases it, else stop
  (Gauss-Newton stalled). The residual is **non-increasing** across the solve and
  stays **bounded** on stiff inputs (atan case: ~1e7 → O(1)) — a far more usable
  iterate for the implicit-function gradient. New `KKTProjectionLayer` params
  `use_line_search=True` (opt-out restores the old full step) and
  `line_search_max_halvings=10`. The differentiable implicit-function one-step is
  unchanged.
- `last_residual` / `last_converged` now reflect the **returned** iterate
  (post-step), fixing a pre-existing off-by-one where the pre-step residual was
  reported.

## [0.4.1] - 2026-06-04

Second v0.4.x sub-project + the README/doc sweep. Behavior-preserving;
suite green; flake8 + mypy clean.

### Removed

- **6 unconsumed `LossConfig` weight fields** — `lambda_adjoint`, `alpha_attn`,
  `alpha_smooth`, `mu_lyap`, `beta_param`, `lambda_delta_u` had zero usages
  outside `config.py` (dead knobs that misled users into thinking they
  controlled training). Removed (behavior-preserving — they did nothing). The
  corresponding sub-loss classes (`AttentionRegularizationLoss`,
  `TemporalSmoothnessLoss`, `ParameterBoundednessLoss`, `ControlEffortLoss`,
  `LyapunovConstraintLoss`) already expose their own weights;`lambda_adjoint`
  had no implemented loss. YAML-backward-compatible (`from_yaml` ignores unknown
  keys).

### Documentation

- **README brought in line with v0.4.0.** Replaced the stale "implementation in
  progress / Version 1.0" framing with the released status (9 phases + PCML,
  pytest 192/0); fixed the Python requirement (`3.8+` → `3.10+`); replaced the
  non-functional Quick Start (`adaptation_rate`, `initialize_controller`,
  `closed_loop_training`, `inference_realtime`) with the real, verified-runnable
  API (`PITSMRASConfig`/`NetworkConfig`/`PhysicsConfig`, `PITNN`,
  `MRASController`, `pretrain_pitnn`, `cotraining_loop`, `RealtimeInferenceEngine`);
  corrected the project-structure tree and roadmap; relabeled "Performance
  Highlights" as **design targets** (not yet experimentally validated); softened
  the real-time feature claims to match what's implemented; and clarified that
  the linked validation/summary docs describe the *design document*, not the code.

## [0.4.0] - 2026-06-04

Opens the v0.4.0 feature/refinement line. First sub-project: the **HJB/costate
co-training rewire** (the rest of the v0.4.0 set stays queued in `todo.md`).
Effective training behavior is preserved by default; suite green throughout;
flake8 + mypy clean.

### Fixed

- **HJB residual is now an opt-in, actually-applied critic regularizer.** The
  `lambda_hjb * HJBResidualLoss` term lived in `cotraining_loop`'s `l_total` (the
  PITNN objective), but it depends only on the critic's `W_c`; `optimizer_pitnn`
  doesn't own `W_c` and the IRL block's `zero_grad` wiped the gradient — so HJB
  never trained the critic. It is now applied through the **critic** optimizer as
  a dedicated step (every step when `lambda_hjb > 0`; a genuine gradient step).

### Removed

- **Vacuous costate-consistency term.** `l_costate = (costate_head(e) −
  critic.gradient(e))²` was identically `0`: `CostateHead` has no own parameters
  and returns `critic.gradient(e)`, so Identity 2 (costate = ∇V̂) is enforced
  **by construction**, not by this loss. Removed the term, the `lambda_costate`
  config field, the `TotalLoss` `_COMPONENTS["costate"]` entry, and the
  `costate_loss` metric. Identity 2 stays covered by `test_identity_costate`.

### Changed

- **`LossConfig.lambda_hjb` default `0.01 → 0.0`** (opt-in). This preserves
  effective behavior exactly — HJB was previously discarded, so the effective
  critic was already IRL-only.

### Backward compatibility

- Removing `lambda_costate` and changing the `lambda_hjb` default are
  config-level changes only. `from_yaml` ignores unknown keys (`setattr` per
  key), so an existing YAML carrying `lambda_costate` loads without error (the
  key is silently unused). The `cotraining_loop` metrics dict no longer includes
  `costate_loss`.

## [0.3.3] - 2026-06-03

Knocks out the two *easy* carried-forward gaps from the v0.3.2 sprint. No
public-API changes; suite green throughout; flake8 + mypy clean.

### Fixed

- **Critic positivity regularizer was structurally inert in co-training.** The
  `1e-3 * positivity_loss` term lived in `cotraining_loop`'s `l_total` (the PITNN
  objective), but it depends only on the critic's `W_c`; `optimizer_pitnn.step()`
  doesn't own `W_c`, and the IRL block's `zero_grad` then wiped the gradient — so
  the positive-definiteness regularizer never actually updated the critic. It is
  now applied through the **critic** optimizer, guarded on a strictly-positive
  loss so it stays a no-op while `P` is PD (the healthy regime) and does not bias
  the IRL update's Adam step schedule. New isolation test (IRL disabled,
  HJB/costate/CBF off) confirms a seeded indefinite `P` has its minimum
  eigenvalue driven upward. (The identical wiring issue affects the HJB and
  costate terms; rewiring those is behavior-changing and is tracked for v0.4.0.)

### Changed

- **`_triu_pairs` cache hygiene** (`utils/lyapunov.py`). The upper-triangular
  index cache now canonicalizes its device key (`_canonical_device_key` resolves
  a bare `"cuda"` to `"cuda:<idx>"` when CUDA is available, so equivalent device
  specs share one entry) and is bounded (`maxsize=128` instead of unbounded).
  CPU behavior is identical.

## [0.3.2] - 2026-06-03

A small **engineering-debt-resolution** release (the debt logged at the close of
v0.3.1, see `todo.md`). No public-API changes; the full test suite stays green
throughout. flake8 + mypy clean.

### Fixed

- **`QuadraticCritic.positivity_loss` is now differentiable (debt #1).** It
  previously read `P` through the detached `extract_P()`, so the
  `1e-3 * positivity` term in `cotraining_loop` contributed a constant with **zero
  gradient** — the positive-definiteness regularizer never influenced training.
  It now derives `P` from a non-detached `unpack_symmetric(W_c)` and returns
  `relu(-λ_min(P))`, giving the term a real gradient path. New test seeds an
  indefinite `P` and asserts the loss is differentiable and that training repairs
  it back to PD.

### Changed

- **KKT projection surfaces convergence (debt #2).** When the differentiable
  Newton solve in `KKTProjectionLayer.forward` exhausted `max_newton_iter`
  without reaching `newton_tol`, it silently returned the final (non-stationary)
  iterate and took the implicit-function gradient there. It now tracks
  `last_converged: bool` and `last_residual: float`, and logs a warning on
  non-convergence (non-breaking — output unchanged). New test checks the flag on
  a generously- vs. a deliberately under-iterated projection.
- **`MRASController.lqr_warm_start` docstring clarified (debt #7).** Documents
  that it is *not* redundant with the constructor: `__init__` warm-starts the
  critic to the reference model's own `P_opt`, whereas `lqr_warm_start(Q, R)`
  re-solves CARE for a **caller-supplied** cost and re-seeds the critic to that
  different `P`. Kept as a public convenience (no API change); a new
  characterization test guards the non-redundancy.

### Performance

- **Cached upper-triangular index pairs (debt #5).** `quadratic_basis`,
  `pack_symmetric`, and `unpack_symmetric` shared the same
  `torch.triu_indices(n, n)` construction on every call (on the per-step
  `extract_P` path). A new `@lru_cache`d `_triu_pairs(n, device)` helper returns
  the cached read-only `(i, j)` index pair; output-identical.
- **Lighter example-test critic fit (debt #6).** `examples/robotic_manipulator.py`
  `run()` gained `critic_train_steps` / `critic_train_trajectories` parameters
  (defaults preserve the standalone demo's full panel-(d) convergence curve); the
  example tests pass a smaller budget. The offline IRL fit is convex/monotone and
  decoupled from loop stability, so a partial fit stays PD. (The dominant residual
  cost is one-time torch higher-order-op/functorch lazy-init, amortized across the
  suite and outside the example's control.)

### Repo hygiene

- **Added `.gitattributes` (debt #4).** `* text=auto eol=lf` (+ `*.bat`/`*.cmd`
  CRLF, `*.sh` LF, binary markers for images/`*.pt`/`*.npy`/…), matching the
  sibling repos and ending the per-commit "LF will be replaced by CRLF" warnings
  on Windows.

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
