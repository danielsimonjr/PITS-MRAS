# TODO

Working tracker for the **PCML** (Physics-Constrained Machine Learning) effort —
adding soft + hard physics-constraint enforcement (Patel et al. 2022; DAE-HardNet,
arXiv:2512.05881) on top of the v0.2.0 PITS-MRAS framework. See
[CHANGELOG.md](CHANGELOG.md) for landed work and `docs/PITS-MRAS — PCML Addendum.md`
for the spec.

## Done

- **Pre-PCML audit remediation** (faithfulness to Implementation Plan §3). 146/146
  tests green, flake8/mypy clean. Four fixes — see CHANGELOG `[Unreleased]`:
  - #1 CostateHead ½-factor (`-2Ke` → `-Ke`); `half_grad` param.
  - #4 Port-Hamiltonian dissipation made pH-consistent (p-block, `∂H/∂p`); energy
    residual vanishes by construction.
  - #2 MRAS feedback routed through the costate head (critic warm-started to
    `P_opt`); Identities 2 & 4 now live in the control loop.
  - #3 `mras_regressor`, `dpg_action_value_gradient`, `dpg_actor_step` (DPG actor).

## Done — PCML feature (TDD, faithful to DAE-HardNet) — released v0.3.0

- [x] **Constraints library** `src/pits_mras/constraints/{base,mechanical,thermal}.py`
  — `PhysicsConstraints` ABC + `ConstraintSpec`, `MechanicalDAE`, `HeatConductionDAE`.
- [x] **PCML core** `src/pits_mras/models/pcml.py` — `SoftPCMLLoss`,
  `TaylorNeighborhoodApproximation`, `KKTProjectionLayer` (Fischer-Burmeister +
  differentiable Newton, implicit-function-theorem gradient), `PCMLModule`
  (soft/hard dynamic activation at `η`).
- [x] **Lagrangian head** `src/pits_mras/models/lagrangian_head.py`.
- [x] **Integration** — `PCMLConfig`; `lam_hat` head on `pitnn.py`; `pcml`
  component in `TotalLoss`; opt-in `pcml_module` hooks in `cotrain.py` (dynamic
  activation) and `realtime.py` (projection bypass). All opt-in / backward-compat.
- [x] **Tests** — `test_pcml_constraints.py`, `test_pcml_soft.py`,
  `test_pcml_hard.py`, `test_pcml_integration.py` (full suite 174/174).
- [x] **Docs + version** — PCML section in `FINAL_SUMMARY` + README; `__init__`
  exports; bumped to v0.3.0.

## Done — architecture tooling + docs (2026-06-02)

- [x] Copied `tools/` from nanoclaw (chunking-for-files, compress-for-context as-is).
- [x] Ported `create-dependency-graph` to a standalone **Python** tool
  (`tools/create-dependency-graph/create_dependency_graph.py`, 10 unit tests);
  parses Python imports/exports/barrels/`TYPE_CHECKING`/cycles/unused/coverage.
- [x] Generated `docs/architecture/` reports (0 circular deps, 0 unused) and
  wrote the 5 docs (OVERVIEW, ARCHITECTURE [moved+refreshed], COMPONENTS, API,
  DATAFLOW); README links updated.

## Resolved 2026-06-02 (former deferred items)

- [x] **#1 coordinate-bearing PCML**: added `examples/pcml_heat_diffusion.py` —
  hard PCML on the 1-D heat equation with genuine `(x, t, ∂)` (autodiff
  derivatives); soft training reduces the residual and the KKT projection drives
  the violation to ~0. (The MRAS control loop has no spatial coordinates, so its
  PCML hook's zero `x`/`t` are correct-by-domain, not a placeholder bug.)
- [x] **#2 lightly-tested PCML paths**: added tests for `order=2` Taylor and the
  holonomic `MechanicalDAE`. **Found + fixed a latent bug**: `MechanicalDAE`'s
  `ConstraintSpec` counts didn't match the residual vector widths (EOM is
  `n_joints`-wide; `n_differential` was 1), which would malform the KKT
  projection on mechanical systems. Spec now reports true widths; KKT projection
  on a holonomic `MechanicalDAE` verified (violation < 1e-3).
- [x] **#3a AV-CBF non-vacuous**: lane-hold-under-gust scenario with a tight
  ellipsoid; the CBF now engages (~11% under the gust) and bounds the departure
  / safe-set violation, with a CBF-activation panel. Honestly framed as a
  minimally-invasive backstop (the near-LQR-optimal nominal needs little help).
- [x] **#3b manipulator critic training**: added `train_irl_critic_gd` (offline
  gradient IRL fit, decoupled from control-loop stability); the demo perturbs
  the critic and trains it back, so panel (d) is a real convergence curve
  (rel-err → ~1e-3). Also added a `critic_convergence` metric to `cotraining_loop`.

## Done — v0.3.1 (2026-06-03): minimize / simplify / optimize

Behavior- and API-preserving pass (Approach 2; spec in
`docs/superpowers/specs/2026-06-03-v0.3.1-simplification-design.md`):

- [x] **A** — consolidated the quadratic-basis convention into
  `utils/lyapunov.py` (`pack_symmetric`/`unpack_symmetric`); critic + IRL trainer
  delegate (3 hand-rolled loops removed).
- [x] **C** — dropped the redundant `f`/`H` keys from `PITNN.forward`.
- [x] **D** — measured perf: reuse the converged Newton iterate in the KKT
  projection one-step (output-identical, ~9% faster); D2 vectorization landed
  with A; D3 dropped (sub-ms noise).
- [x] Version → 0.3.1; dependency graph regenerated; `docs/architecture` stat
  references synced; CHANGELOG `[0.3.1]`.
- Out of the safe pass: **B** (6 dead `LossConfig` fields) → logged as debt for
  v0.3.2; **E** (`parallel.py`) → v0.4.0 capability.

## Engineering debt (target v0.3.2 — a debt-resolution release)

Discovered / pre-existing issues to plan + resolve in v0.3.2. None is currently
causing test failures; each is grounded below.

### Correctness / efficacy

1. **`positivity_loss` is a gradient no-op** (verified 2026-06-03).
   `QuadraticCritic.extract_P()` calls `.detach()`, so `positivity_loss()` has
   `requires_grad=False` / no `grad_fn`; the `1e-3 * positivity` term in
   `cotraining_loop` adds a constant with **zero gradient** — the
   positive-definiteness regularizer never actually influences training (P̂
   positivity is currently held only by the warm-start + IRL fit). *Fix:* give
   the term a differentiable path (e.g. eigenvalues of a non-detached
   `unpack_symmetric(W_c)`, or a differentiable `relu(eps - λ_min)` surrogate);
   add a test asserting a non-zero gradient and that training repairs a seeded
   indefinite `P`.
2. **KKT projection silent non-convergence.** When Newton exhausts
   `max_newton_iter` without hitting `newton_tol`,
   `KKTProjectionLayer.forward` still returns the final iterate and takes the
   implicit-function gradient at a **non-stationary** point (approximate), with
   no signal. *Fix:* surface a convergence flag / max-residual (return or
   warning); consider a damped / line-search Newton step for robustness; test
   the flag on a deliberately under-iterated projection.

### Repo hygiene

4. **No `.gitattributes`** → every Windows commit warns "LF will be replaced by
   CRLF" and risks mixed line endings. Add `* text=auto eol=lf` (+ `*.bat` /
   `*.cmd eol=crlf`, `*.sh eol=lf`), matching the sibling repos.

### Low priority / watch

5. **Tiny-matrix basis ops** — `pack/unpack_symmetric` via `torch.triu_indices`
   is ~100 µs slower than a 3-element loop at `n=2` (overhead-bound; a clear win
   for `n≥3`). Negligible vs a training step; revisit (cache the indices) only if
   profiling ever flags the per-step `extract_P` path.
6. **Slow example integration tests** — the example `run()` tests are heavy
   (manipulator critic training + KKT autograd). Fine now; candidate for `pytest`
   markers / smaller step budgets if CI wall-clock becomes a concern.
7. **Redundant `lqr_warm_start`** — `MRASController.__init__` already warm-starts
   the critic to `P_opt`; `lqr_warm_start` re-solves CARE and is not called by
   the loops. Decide: keep as a public convenience or drop (API change).

## v0.4.0 (next version) — features / refinements / new capacities

- **Dead `LossConfig` fields → wire-or-remove** (bumped from v0.3.2 debt): the 6
  unconsumed fields `lambda_adjoint`, `alpha_attn`, `alpha_smooth`, `mu_lyap`,
  `beta_param`, `lambda_delta_u`. Decide per field: **wire** into the
  corresponding sub-loss (adjoint-dynamics residual, attention entropy /
  smoothness, Lyapunov-decay rate, parameter-boundedness, control-rate penalty)
  — a capability addition — or **remove** (a public `PITSMRASConfig`/`from_yaml`
  change, hence a minor bump). Belongs with the feature work.
- **H∞ disturbance/adversary head (gap G1, Blueprint Connection 7).** A new
  adversary network head, a Game Algebraic Riccati Equation (GARE) solver
  (`solve_gare`, not yet implemented), and the robust-control / worst-case
  min-max training loop. The Blueprint describes it; the Implementation Plan
  built critic/costate/CBF as the three concrete heads. Major capability.
- **Complete `ParallelInferenceEngine`** (`inference/parallel.py`) from the
  honest threaded skeleton to a hardened multi-rate (1 kHz / 100 Hz / 10 Hz)
  deployment with the double-buffered critic swap.
- **Higher-fidelity example plants** — replace the linear reference-model
  surrogates with a nonlinear rigid-body manipulator, a bicycle/tyre AV model,
  and an RC building-thermal network.

## Notes / decisions

- The KKT hard-projection layer is research-grade; implement faithfully to
  DAE-HardNet (Eq. 2/3/12/13/15), fixing the Addendum's draft differentiability
  issues (detach-then-require-grad, Jacobian construction). Cross-checked against the
  primary PDF (`Misc/DAE-HardNet — ...pdf`) and Patel 2022 (`Misc/Physics
  Constrained Learning ...pdf`).
- Re-run gates after structural changes: `python -m pytest -q`,
  `flake8 src tests --max-line-length=100 --ignore=E203,W503`,
  `mypy src/pits_mras --ignore-missing-imports`. (`black`/`isort` not installed
  locally; CI enforces them.)
