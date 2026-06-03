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

## Done — engineering debt resolved (v0.3.2 sprint, 2026-06-03)

The debt logged at the close of v0.3.1, all resolved via the dev-workflow (TDD,
suite green throughout, flake8 + mypy clean). No public-API changes.

### Correctness / efficacy

- [x] **#1 `positivity_loss` gradient no-op** (commit 35ebf3c). Now derives `P`
  from a non-detached `unpack_symmetric(W_c)` and returns `relu(-λ_min(P))`, so
  the `1e-3 * positivity` term in `cotraining_loop` has a real gradient path.
  Test seeds an indefinite `P` and asserts differentiability + training repairs
  it to PD. (`extract_P` stays detached for read-only callers.)
- [x] **#2 KKT projection silent non-convergence** (commit 11251ad).
  `KKTProjectionLayer` now tracks `last_converged` / `last_residual` and logs a
  warning when Newton exhausts `max_newton_iter` without hitting `newton_tol`
  (non-breaking — output unchanged). Test checks the flag on a generously- vs.
  under-iterated projection.

### Repo hygiene

- [x] **#4 No `.gitattributes`** (commit a73fb52). Added `* text=auto eol=lf`
  (+ `*.bat`/`*.cmd` CRLF, `*.sh` LF, binary markers); `git add --renormalize`
  confirmed the index was already LF.

### Low priority / watch

- [x] **#5 Tiny-matrix basis ops** — added an `@lru_cache`d `_triu_pairs(n,
  device)` helper in `utils/lyapunov.py`; `quadratic_basis` / `pack_symmetric` /
  `unpack_symmetric` reuse the cached read-only `(i, j)` index pair instead of
  rebuilding `torch.triu_indices` each call. Output-identical.
- [x] **#6 Slow example integration tests** — measured: the manipulator example
  cost is dominated by **one-time torch higher-order-op/functorch lazy-init**
  (~15 s the first time it runs in a process; ~6 s amortized in the full suite),
  *not* the example's own compute. Still parameterized the IRL fit
  (`critic_train_steps` / `critic_train_trajectories` on `run()`, defaults
  preserve the demo) so tests pass a lighter budget — a genuine if modest win.
  The residual lazy-init cost is amortized across the suite and outside our
  control; no `pytest` markers needed.
- [x] **#7 Redundant `lqr_warm_start`** — decided **keep + document**: it is
  *not* redundant with the constructor (`__init__` warm-starts to the ref model's
  own `P_opt`; `lqr_warm_start(Q, R)` re-solves CARE for a caller-supplied cost).
  Docstring clarified; characterization test guards the non-redundancy. No API
  change.

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
