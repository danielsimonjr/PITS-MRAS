# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2026-06-06

Closes the last open feature follow-on (sequence-`PITNN` → H∞ min-max wiring) and
homes the accumulated tooling/docs work: de-versioned architecture docs + CDG
reports, the `LatentModel` re-export, the COMPONENTS catalog sweep, the living-doc
refresh, and the dev-extras toolchain fix. Suite green (386); ruff + mypy clean;
graph 0 circular / 0 unused.

### Added

- **`pitnn_one_step` + `hinf_minmax_from_pitnn`** (`training/hinf_minmax.py`,
  re-exported from `pits_mras.training`) — wire the full sequence-`PITNN` into the
  H∞ neural min-max loop. `pitnn_one_step(pitnn, history=None)` collapses the
  sequence model into a one-step `f(x, u) -> xdot` (documented operating-point /
  history convention: fixed history context, varies the current `[q, p]` state +
  control, first-order tangent about tracking error `e = 0`);
  `hinf_minmax_from_pitnn(...)` runs the existing min-max loop on it. A learned
  PITNN is nonlinear, so GARE-oracle recovery is not expected — verification is
  finiteness / shape / differentiability / end-to-end. This closes the last open
  feature follow-on.
- **`linearize_dynamics` gains an opt-in `backend="autograd"`** (default
  `"jacrev"` unchanged). The `autograd` backend
  (`torch.autograd.functional.jacobian`) composes with dynamics callables that run
  `torch.autograd.grad` internally — required for the PITNN adapter, whose
  port-Hamiltonian decoder differentiates a learned Hamiltonian inside `forward`
  (which `torch.func.jacrev` forbids).

### Tooling

- **`create-dependency-graph` no longer stamps version/date into its reports.** The
  generated `DEPENDENCY_GRAPH.md` / `TEST_COVERAGE.md` / `unused-analysis.md` and the
  JSON/YAML no longer carry `**Version**`/`**Last Updated**`/`**Generated**`/
  `lastUpdated`/`generatedAt`/`v`/`d` fields — so the architecture reports are
  version/date-free snapshots (matching the narrative docs) and fully reproducible
  (no per-day churn). The tool still reads the project name; version/date live in the
  CHANGELOG.
- **`setup.py` dev extras now install Ruff**, not the retired flake8/black/isort —
  `pip install -e ".[dev]"` matches the actual toolchain (`ruff==0.8.1` + mypy +
  pytest + pytest-cov).

### Fixed

- **`LatentModel` (TD-MPC2 planner interface) is now re-exported** from
  `pits_mras.models`. It was a public `Protocol` used only as an internal type
  annotation, so the dependency graph (correctly) flagged it as an unused export;
  re-exporting it makes the typed contract importable for custom world models and
  restores a truthful **0 unused exports**. (The prior "0 unused" in v0.7.0/v0.8.0
  notes was a carried-forward figure — it had actually been 1 since `LatentModel`
  was introduced.)

### Docs

- **Living user-facing docs refreshed to the current state.** `README.md`
  (version badge, capability list, project tree, CI/toolchain), `src/README.md`
  (present-tense module inventory), `CONTRIBUTING.md` / `GITHUB_SETUP.md` (Ruff
  workflow, not flake8/black/isort), and `PROJECT_SETUP_SUMMARY.md` de-versioned.
  Stale version stamps and hardcoded test counts removed (non-churning phrasing).
- **`COMPONENTS.md` per-module catalog brought current.** Its detailed sections
  (`utils`, `models`, `losses`, `controllers`, `training`) omitted every module
  added in the two feature sprints; extended them — grounded in the source
  docstrings — to cover `adversary`, `koopman`, `sac`, `tdmpc`, `generic`,
  `diagnostics`, `uq`, `linearization`, `adaptive_weighting`, `koopman_control`,
  and the H∞ min-max / SAC / TD-MPC2 trainers, and added a new `data/` section
  (`TrajectoryDataset`, `generate_synthetic_trajectories`, `make_dataloader`).
  Present-tense, version/date-free.
- **Architecture docs de-versioned to read as a present-tense design snapshot.**
  Removed version numbers, dates, and "what changed when" language (e.g. `NEW`,
  `RESOLVED vX`, `shipped in vX`, per-version Status blocks, the resolved-gap rows,
  the `Version:` stat field) from the hand-written architecture docs
  (`ARCHITECTURE.md`, `OVERVIEW.md`, `COMPONENTS.md`, `API.md`, `DATAFLOW.md`) and
  rewrote the affected sections in present tense. Version/date history lives here
  in the CHANGELOG and in the generated dependency-graph/test-coverage reports;
  the design docs now describe only the current state (current counts retained).
  The generated reports were not hand-edited.

## [0.8.0] - 2026-06-06

Gap-closure sprint — **GENERIC/GFINN thermodynamic decoder** (the final gap; the
thermodynamically-consistent generalization of the port-Hamiltonian decoder).
Suite green (380); ruff + mypy clean.

### Added

- **`GFINNDecoder`** (`models/generic.py`, re-exported from `pits_mras.models`) —
  a GENERIC-formalism decoder (Zhang/Shin/Karniadakis 2022) with learned scalar
  potentials `E(z)` (energy), `S(z)` (entropy) and operators `L(z)` (skew),
  `M(z)` (PSD), giving `ż = L∇E + M∇S`. The thermodynamic laws hold **by
  construction**: `L` is built as `Σ(âb̂ᵀ − b̂âᵀ)` with `â, b̂` projected
  orthogonal to `∇S` (so `L∇S = 0` and `Lᵀ = −L`); `M = Σ d̂d̂ᵀ` with `d̂`
  projected orthogonal to `∇E` (so `M∇E = 0`, `M ⪰ 0`). Hence `dE/dt = 0`
  (first law) and `dS/dt = ∇SᵀM∇S ≥ 0` (second law). Verified to ≤1e-5 residuals
  (skew, PSD, degeneracy, energy conservation, entropy production). 9 new tests.

## [0.7.0] - 2026-06-06

Gap-closure sprint — **Connection 9: TD-MPC2 / learned-model planning** (the
second of the two Blueprint connections that previously had no module). A
self-contained latent world model + sampling-based planner. Suite green (371);
ruff + mypy clean.

### Added

- **TD-MPC2 world model** (`models/tdmpc.py`, re-exported from `pits_mras.models`):
  - `WorldModel` — latent `encode` / `next` / `reward` / `value` / `Q` heads
    (Hansen et al. 2024) over a shared latent space.
  - `MPPIPlanner` — gradient-free sampling-based MPC in latent space: samples
    action sequences, rolls them through the learned model, scores by discounted
    reward + terminal value, and refits a Gaussian to the MPPI-weighted elites.
    Consumes any object exposing `next`/`reward`/`value` (the `LatentModel`
    Protocol).
- **`tdmpc_update`** (`training/tdmpc.py`, re-exported from `pits_mras.training`)
  — joint world-model loss (latent consistency + reward MSE + TD value).
  - Verified by a falsifiable planner test: with the `WorldModel` set to the
    ground-truth of a linear-quadratic problem, the MPPI planner's first action
    recovers the LQR-optimal `a* = -K*z₀` to relative error 0.13. 9 new tests.

## [0.6.0] - 2026-06-06

Gap-closure sprint — **Connection 5: SAC / max-entropy RL** (one of the two
Blueprint connections that previously had no module). A self-contained, additive
Soft Actor-Critic. Suite green (362); ruff + mypy clean.

### Added

- **SAC policy + critics** (`models/sac.py`, re-exported from `pits_mras.models`):
  - `GaussianPolicy` — tanh-squashed reparameterized Gaussian actor with the exact
    log-prob squash correction; deterministic `mean` path; configurable
    `action_scale` / `log_std_bounds`.
  - `TwinQCritic` — twin `Q(s,a)` networks with a `q_min` helper.
- **`SACTrainer`** (`training/sac.py`, re-exported from `pits_mras.training`) —
  full Soft Actor-Critic (Haarnoja et al. 2018) with **automatic entropy
  temperature** (`log_alpha` toward target entropy `-action_dim`), twin-Q targets,
  and Polyak soft updates. `update(batch)` implements the standard critic/actor/
  temperature losses and returns finite diagnostics + `alpha`.
  - Verified by a falsifiable learning sanity (single-step bandit, reward
    `-‖a−target‖²`): the greedy policy action converges from 0.73 to 0.015 of the
    target. 11 new tests.

## [0.5.5] - 2026-06-05

Gap-closure sprint — integration item **#6** (wire the H∞ neural min-max loop to
learned/analytic dynamics). Additive; suite green (351); ruff + mypy clean.

### Added

- **`linearize_dynamics(dynamics_fn, x0, u0) -> (A, B)`** (`utils/linearization.py`)
  — first-order Jacobian linearization of any continuous-time dynamics callable
  `f(x, u) -> xdot` about an operating point, via `torch.func.jacrev` (exact for
  affine `f`).
- **`hinf_minmax_from_dynamics(dynamics_fn, x0, u0, Q, R, gamma, D=None, **kwargs)`**
  (`training/hinf_minmax.py`, re-exported from `pits_mras.training`) — the bridge
  that linearizes a learned/analytic dynamics callable at `(x0, u0)` and runs the
  existing `hinf_minmax_train` on the resulting `(A, B)` (returns the metrics plus
  the extracted `A`, `B`). Lets the robust min-max run on a Koopman `latent_step`,
  a plant `f`, or any one-step dynamics. Verified: oracle recovery on linear
  dynamics + a Koopman-`latent_step` bridge smoke test. 7 new tests.
  `hinf_minmax_train` itself is unchanged (the un-skipped tight recovery test
  stays green). *Note:* collapsing the full sequence-`PITNN` into a one-step
  `f(x,u)` (operating-point/history choices) is left as a documented follow-on.

## [0.5.4] - 2026-06-05

Gap-closure sprint — integration item **#5** (wire deep Koopman lifting into the
control loop). Additive; suite green (344); ruff + mypy clean.

### Added

- **`KoopmanLQRController`** (`controllers/koopman_control.py`, importable via
  `from pits_mras.controllers.koopman_control import KoopmanLQRController`) — closes
  the loop on Koopman-lifted coordinates: reads the learned latent linear system
  `(A_z, B_z)` from a (frozen) `KoopmanLiftingModel`, embeds the state-cost `Q`
  into the lifted coordinates (`q_latent` override supported), solves
  `P_z, K_z = solve_care(A_z, B_z, Q_z, R)` at construction, and produces
  `u = -(encode(x) - encode(x_ref)) @ K_zᵀ`. This realizes the v0.4.14 Koopman
  model's purpose — applying the verifiable linear core (CARE) on the lifted space.
  Verified by oracle recovery (`K_z` equals a direct `solve_care`; latent closed
  loop `A_z - B_z K_z` Hurwitz) + lifted-error decay. 12 new tests.

## [0.5.3] - 2026-06-05

Hotfix for v0.5.2: the `src/pits_mras/data/` source package was silently excluded
by the `data/` rule in `.gitignore`, so v0.5.2 shipped its tests but **not** the
module they import — a fresh checkout (and CI) would fail to import `pits_mras.data`.

### Fixed

- **Track the first-party `data/` package.** Added `.gitignore` exceptions
  (`!src/pits_mras/data/`, `!src/pits_mras/data/**`) so the source package is
  committed; the `data/` / `datasets/` ignore rules still cover real dataset
  directories. `src/pits_mras/data/__init__.py` and `trajectory.py` are now in the
  repo. (Caught by a git-tracking check; local tests had passed because the files
  existed on disk.)

## [0.5.2] - 2026-06-05

Gap-closure sprint — gap **G7** (`data/` dataset/loader module). Additive + opt-in;
suite green (332); ruff + mypy clean.

### Added

- **`pits_mras.data` package** (`data/trajectory.py`) — a reusable, opt-in
  trajectory-data layer (torch/numpy only):
  - `TrajectoryDataset(torch.utils.data.Dataset)` — windowed `(state_hist,
    control_hist, state, control, next_state)` samples from one or many
    trajectories, with shape validation.
  - `generate_synthetic_trajectories(...)` — seedable forward-Euler rollout of the
    same plant the inline co-training code uses (factored out for reuse).
  - `make_dataloader(...)` — batching helper with a PITNN-shaped collate.
- **Opt-in dataset path in `pretrain_pitnn`** (`dataset=None` default). When
  `None`, behaviour is unchanged (the default path is byte-identical — verified;
  `cotrain` deliberately left untouched to protect its characterization lock).
  When supplied, pre-training draws windowed batches from the dataset.
  15 new tests. `data/` is imported lazily on the opt-in path only, so
  `import pits_mras` and the public surface are unchanged.

## [0.5.1] - 2026-06-05

Gap-closure sprint — gap **G8** (MIMO control input). Suite green (317);
ruff + mypy clean.

### Fixed

- **G8 — MIMO control input generalized in the port-Hamiltonian decoder.** The
  control term was `f_ctrl = B(x_p)·sum(u)` (a scalar collapse of multi-input
  control). It is now a true input-matrix product `f_ctrl = B(x_p) @ u`:
  `B_net` emits `[batch, 2·n_q, control_dim]` and `f_ctrl = bmm(B, u)`. `control_dim`
  is threaded into `PortHamiltonianDecoder` (default 1) and wired from `PITNN`.
  **`control_dim=1` is exactly preserved** (mathematically identical to the old
  path; locked by a characterization test), so the shipped single-input examples
  are unaffected; multi-coordinate plants now get genuine per-channel control.
  4 new tests (characterization + MIMO correctness).

### Docs

- Marked the now-stale ARCHITECTURE.md notes resolved: the H∞ neural min-max loop
  (shipped v0.5.0) and gap G8 (this release).

## [0.5.0] - 2026-06-05

**H∞ neural adversarial min-max training loop** (ROADMAP #1) — the headline new
feature line on top of the v0.4.5 analytic H∞ core. Closes the long-planned
v0.5.0 milestone. Additive (the analytic core is untouched); suite green (312
passed, 1 documented skip); ruff + mypy clean.

### Added

- **`NeuralAdversary`** (`models/adversary.py`, re-exported from `pits_mras.models`)
  — a *learned* disturbance policy `w = π(e)` (Tanh-MLP, small-init so the
  disturbance starts weak), the trainable counterpart to the analytic
  `AdversaryHead` (`w* = γ⁻²DᵀPe`).
- **`hji_residual`** and **`hinf_minmax_train`** (`training/hinf_minmax.py`,
  re-exported from `pits_mras.training`) — a three-network actor–critic–adversary
  ADP loop solving the Hamilton–Jacobi–Isaacs equation as a min-max game:
  - HJI residual `ρ(e) = eᵀQe + uᵀRu − γ²‖w‖² + ∇V̂·(Ae + Bu + Dw)` with
    `u = −½R⁻¹Bᵀ∇V̂` (costate head) and `w = NeuralAdversary(e)`.
  - Critic minimizes `E[ρ²]` (+ positivity); the adversary ascends (maximizes)
    `E[ρ]`; the protagonist is the implicit slow player. Two-timescale learning
    rates (`adv_lr ≥ critic_lr`, both fast vs. the protagonist; Borkar / TTUR)
    stabilize the min-max.
  - **Verified against the analytic GARE oracle** (`solve_gare`): on a linear
    plant the trained critic/gain/adversary recover `(P*, K*, L*)` tightly
    (`‖P̂−P*‖/‖P*‖ ~1e-5` at γ=5), and a training-free objective-correctness check
    gives `max|ρ| < 1e-3` with the analytic value+heads. `γ→∞` recovers the
    LQR/CARE solution. 8 new tests (1 tight-equality variant skipped by design —
    the trend test is the non-flaky equivalent).

### Notes

- Control-loop integration with the deep Koopman lifting (v0.4.14) — running the
  neural min-max on lifted coordinates — is a natural follow-on, not included.

## [0.4.14] - 2026-06-05

Sprint item ROADMAP #2 (deep Koopman lifting model). Additive capability — not
wired into the control loop (integration is a documented follow-on). Suite green
(304); ruff + mypy clean.

### Added

- **`KoopmanLiftingModel`** + **`koopman_loss`** (`models/koopman.py`, re-exported
  from `pits_mras.models`) — a deep Koopman model (Lusch et al. 2018) that lifts
  the nonlinear state into a latent space with *learnable linear* dynamics
  `(A_z, B_z)`, so the existing linear core (quadratic critic, `solve_care` /
  `solve_gare`, CLF-CBF) can be applied on lifted coordinates.
  - `encode` / `latent_step` (exactly `z @ A_zᵀ + u @ B_zᵀ`) / `decode` /
    `forward`; `latent_matrices()` exposes `(A_z, B_z)` as the bridge to the
    Riccati solvers. `include_state=True` lifts as `[x, ψ(x)]` with exact-slice
    decode (zero reconstruction loss by construction) and a stable identity warm
    start.
  - `koopman_loss` = reconstruction + latent-linearity + state-prediction MSE
    terms (weighted). 15 new tests.
  - Control-loop integration (closing the loop on lifted coordinates via the
    Riccati solvers) is a documented follow-on, not included here.

## [0.4.13] - 2026-06-05

Sprint item ROADMAP #6 (differentiable CARE/GARE via implicit differentiation) —
the enabler for a future neural H∞ min-max loop (#1). Additive; suite green (289);
ruff + mypy clean.

### Added

- **`differentiable_care(A, B, Q, R)`** and **`differentiable_gare(A, B, Q, R,
  gamma, D=None)`** (`utils/lyapunov.py`) — return the stabilizing `P` as a
  differentiable tensor so gradients flow w.r.t. the input matrices, WITHOUT
  differentiating through the scipy solver. Forward solves with the existing
  `solve_care`/`solve_gare` under `no_grad`; backward uses the implicit function
  theorem on the Riccati residual `AᵀP + PA − P M P + Q = 0` (`M = BR⁻¹Bᵀ` for
  CARE, `− γ⁻²DDᵀ` added for GARE): the adjoint state solves
  `A_cl S + S A_clᵀ + sym(∂L/∂P) = 0` (`A_cl = A − MP` Hurwitz at the solution),
  giving `∂L/∂Q = S`, `∂L/∂A = 2PS`, `∂L/∂B`, `∂L/∂R`, `∂L/∂D`. Both backward
  passes are **`torch.autograd.gradcheck`-verified** (float64), including the
  `D`-defaults-to-`B` double-dependence on `B`. 10 new tests
  (`tests/test_differentiable_riccati.py`). Existing `solve_care`/`solve_gare`
  are unchanged.

## [0.4.12] - 2026-06-05

Sprint item ROADMAP #5 (vectorize the KKT constraint-Jacobian). Behaviour-
preserving performance/clarity refactor. Suite green (279); ruff + mypy clean.

### Changed

- **`KKTProjectionLayer` constraint Jacobians now use `torch.func`** (`jacrev` +
  `vmap`) instead of per-constraint Python `autograd.grad` loops. Both loops in
  `_constraints_and_jac` (the `c = [differential; equality]` Jacobian and the
  inequality `g` Jacobian) are replaced by vectorized whole-Jacobian passes via
  pure per-sample helpers (`_eval_c`, `_eval_g`). Numerically identical to the
  old loops (equivalence-tested to `rtol=1e-5`); the projection forward output
  and the implicit-function-theorem gradient are unchanged (golden + reference-
  loop comparison). The surrounding Newton / Fischer-Burmeister / line-search /
  IFT logic is byte-for-byte unchanged. No new dependency (`torch.func` is built
  into torch). 5 new equivalence tests.

## [0.4.11] - 2026-06-05

Sprint item ROADMAP #8 (adaptive / causal loss weighting). Opt-in, default-off;
the v0.4.10 characterization test stays green (behaviour unchanged when disabled).
Suite green (274); ruff + mypy clean.

### Added

- **Adaptive loss-weighting utilities** (`losses/adaptive_weighting.py`, torch-only):
  - `ReLoBRaLo` — Relative Loss Balancing with Random Lookback (Bischof & Kraus,
    arXiv:2110.09813): cheap multi-term balancer using only loss *values* (no
    extra backward passes); weights sum to `num_losses`. Reproducible Bernoulli
    lookback via an optional `torch.Generator` (the no-generator fallback is
    counter-based and never touches global RNG).
  - `causal_weights` — causal training weights `exp(-eps·cumsum_{k<i} residual_k)`
    (Wang, Sankaran, Perdikaris, arXiv:2203.07404) for time-ordered residuals;
    all-weights→1 signals temporal convergence.
  20 new tests.
- **`LossConfig.adaptive_weighting: bool = False`** — opt-in flag. When `True`,
  `cotraining_loop` uses `ReLoBRaLo` to rebalance the PITNN-objective terms
  (physics / PCML / CBF) each step instead of the fixed lambdas; when `False`
  (default) the loop is byte-identical to before.

## [0.4.10] - 2026-06-05

Sprint item ROADMAP #9 (simplicity refactor of the co-training loop). Pure
behaviour-preserving refactor — locked by a new characterization test. Suite
green (254); ruff + mypy clean.

### Changed

- **`cotraining_loop` decomposed into five named helpers** (`_pitnn_objective_step`,
  `_hjb_critic_step`, `_positivity_critic_step`, `_irl_critic_step`,
  `_advance_plant`); the top-level loop now reads as a sequence of named steps.
  Verified behaviour-identical (same ops, order, RNG consumption, optimizer-step
  sequence) by a new `test_cotrain_characterization` golden-value test.
- **CBF weight is now configurable.** The hardcoded `0.1 * L_cbf` in the PITNN
  objective is replaced by `LossConfig.lambda_cbf` (default `0.1`, so behaviour is
  unchanged; YAML-backward-compatible).

### Removed

- **Dead `n_heads` parameter** dropped from `PhysicsInformedAttention` (declared
  but never used in `forward`); the `models/pitnn.py` caller no longer passes it.
  `NetworkConfig.attention_heads` is kept (user-facing / backward-compatible).

### Notes

- The `TotalLoss` registry was **deliberately not** wired into `cotraining_loop`:
  the loop splits losses across two optimizers (PITNN-objective vs critic-side
  IRL/HJB/positivity) and interleaves PCML activation, which `TotalLoss`'s
  single-aggregate model doesn't fit — forcing it would add indirection, not
  remove it. `TotalLoss` remains the aggregator for pre-training / external use.

## [0.4.9] - 2026-06-05

Sprint item ROADMAP #3 (uncertainty quantification), plus a CI fix for the
v0.4.8 ruff-format drift. Suite green (253); ruff + mypy clean.

### Added

- **Uncertainty-quantification utilities** (`utils/uq.py`, importable via
  `from pits_mras.utils.uq import ...`) — torch/numpy only, no new runtime deps:
  - `DeepEnsemble` — epistemic UQ from K member models: `predict_all` (→
    `[K, batch, d]`) and `mean_and_std`.
  - `split_conformal_quantile` — finite-sample split-conformal quantile
    (`ceil((n+1)(1-alpha))/n` order statistic; `+inf` when alpha is too small
    for the calibration size).
  - `conformal_interval` — symmetric prediction interval `(pred-q, pred+q)`.
  - `AdaptiveConformalInference` — online ACI (Gibbs & Candès 2021) for
    non-stationary/time-series coverage. 19 new tests (`tests/test_uq.py`).

### Fixed

- **CI: pin `ruff==0.8.1`** in `requirements-dev.txt`. The previous `ruff>=0.8.0`
  let CI float to a newer ruff whose formatter reformatted 5 files, so
  `ruff format --check` passed locally (0.8.1) but failed in CI — red-lighting
  v0.4.8. The formatter version is now pinned for reproducibility.

## [0.4.8] - 2026-06-05

Sprint item ROADMAP #10 (tooling / typing). Suite green (234); ruff + mypy clean.

### Changed

- **Linter/formatter migrated from flake8 + black + isort to Ruff** (mypy kept).
  `pyproject.toml` now carries `[tool.ruff]` (line-length 100, target py310),
  `[tool.ruff.lint]` (`select = ["E","F","W","I"]`, `ignore = ["E203"]`; the `I`
  rules replace isort), and `[tool.ruff.format]`; `[tool.black]`/`[tool.isort]`
  removed. `setup.cfg` (flake8-only) deleted. CI runs `ruff check .` +
  `ruff format --check .`. `requirements-dev.txt` swaps flake8/black/isort → ruff.
  Configured as a strict drop-in (equivalent ruleset); applying `ruff format`
  reformatted 44 files — pure formatting (token-stream-verified), no behaviour
  change, full suite green afterward.

### Added

- **Fail-loud config validation.** `PITSMRASConfig.from_yaml` now raises
  `ValueError` (naming the offending key) on an unknown top-level section or an
  unknown nested field, instead of silently ignoring it. 3 new tests.
- **jaxtyping shape annotations** (dev-only, under `TYPE_CHECKING`) on the
  tensor helpers in `utils/lyapunov.py` (`quadratic_basis`, `pack_symmetric`,
  `unpack_symmetric`); `jaxtyping` added to `requirements-dev.txt`. Not a runtime
  dependency.

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
