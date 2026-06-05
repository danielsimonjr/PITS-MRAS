# TODO

Working tracker for PITS-MRAS. **Current state: released through v0.4.5** — the
9-phase foundation (v0.2.0), the PCML physics-constraint layer (v0.3.0), the
v0.3.x simplification/debt passes, and the **complete v0.4.x feature line**
(v0.4.0–v0.4.5: HJB/costate rewire, docs sweep, dead-field removal, KKT
line-search Newton, nonlinear example plants, ParallelInferenceEngine hardening,
H∞ adversary core). See [CHANGELOG.md](CHANGELOG.md) for landed work.

See **Open items** below for what's not-yet-done; everything in the dated
release sections further down is DONE.

## Open items (not started)

None blocking — the v0.4.0 goal is fully delivered and all gates are green. These
are genuine optional follow-ons, in rough priority order:

> **2026-06-05 — ROADMAP refreshed.** `docs/ROADMAP.md` was rewritten: the stale,
> fully-implemented 9-phase build plan was removed (CHANGELOG + `docs/architecture/`
> are the durable record), and **10 research-derived improvement proposals** (a
> 4-agent deep-research pass, grounded against the source) were added, grouped by
> capabilities / efficiencies / simplicities, plus a "Known gaps" section (G8 MIMO,
> SAC/Connection 5, TD-MPC2/Connection 9, G7 no `data/` loader). Item 2 below (v0.5.0
> H∞ neural min-max) is now ROADMAP proposal #1. The proposals are candidates, not
> scheduled. See memory `project_pits_mras_improvement_research.md`.

1. **CDG tool — import-parsing accuracy bug** (tooling). The dependency-graph
   tool captures **64 malformed `externalDependencies` import strings**: a
   function-level `from x import y  # noqa: E402` is swallowed *together with the
   following function body* into one giant "import" entry (the logical-line
   joiner mishandles the trailing comment / subsequent code). Pollutes
   `dependency-graph.json` only — the `.md` reports and the headline stats are
   unaffected (which is why it stayed hidden). Bounded, TDD-able; the natural
   follow-on to the v0.4.x-era reproducibility fix (`fix(tools)` c9c2f49).
   Regenerate the graph after fixing.
2. **v0.5.0 — H∞ neural adversarial min-max training loop** (feature). A learned
   adversary *network* + worst-case min-max co-training on top of the v0.4.5
   analytic core (`solve_gare` + `AdversaryHead`). A new feature line — needs its
   own brainstorm → spec → plan → TDD. The only *planned* feature still open.
3. **Verify CI green on GitHub** for the post-v0.4.5 pushes (local gates pass and
   CI runs the same matrix; remote run not yet confirmed this session). Minor.
4. **Tooling `[Unreleased]` CHANGELOG entry** (the CDG reproducibility fix) has no
   tagged home — fine to let it roll into the next release, or cut a patch tag.
   Decision only; no code work.

Deferred/not-actionable now: the "watch" item below (#4 example-test framework
warmup) and the Future roadmap (multi-agent, hierarchical PITS-MRAS, GPU/TPU,
monitoring dashboard).

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

## Done — released v0.4.0 (2026-06-04): HJB/costate co-training rewire

First v0.4.0 sub-project (feature/refinement line), brainstormed → spec
(`docs/superpowers/specs/2026-06-04-v0.4.0-hjb-costate-cotrain-rewire-design.md`)
→ plan → TDD. **HJB** is now an opt-in (default-off, `lambda_hjb>0`) critic
regularizer applied via the critic optimizer (it was in `l_total`, gradient
discarded). **Costate** term removed — brainstorming found it was *identically
0* (`λ̂ ≡ ∇V̂` by construction; not just discarded), so the `lambda_costate`
field, `TotalLoss._COMPONENTS["costate"]`, and the `costate_loss` metric went
too; Identity 2 holds by construction. Version 0.3.3→0.4.0; graph regenerated
(39 files, 5,298 LOC, 116 exports, 0 circular, 0 unused); `docs/architecture` +
`ROADMAP.md` stale cotrain/LossConfig descriptions synced; CHANGELOG `[0.4.0]`;
tagged `v0.4.0`. **The remaining v0.4.0 sub-projects are still queued** (see the
v0.4.0 section): H∞ head, KKT damped Newton, dead `LossConfig` fields,
`ParallelInferenceEngine`, higher-fidelity plants — v0.4.0 is *opened*, not
completed.

## Done — released v0.3.3 (2026-06-03): two easy carried-forward gaps

Knocked out gaps #2 (positivity now applied via the critic optimizer — it was
structurally inert, not just under-weighted) and #3 (`_triu_pairs` device-key
canonicalization + bounded cache) via the dev-workflow (TDD, suite green). The
HJB/costate variant of the same wiring bug is bumped to v0.4.0 (behavior-
changing). Version bumped to 0.3.3; graph regenerated (39 files, 5,302 LOC, 116
exports, 0 circular, 0 unused); `docs/architecture` markers synced; CHANGELOG
`[0.3.3]`; tagged `v0.3.3`. See the "Carried-forward gaps" section for detail.

## Done — released v0.3.2 (2026-06-03): engineering-debt-resolution

The debt logged at the close of v0.3.1, all resolved via the dev-workflow (TDD,
suite green throughout, flake8 + mypy clean). No public-API changes. Version
bumped to 0.3.2; dependency graph regenerated (39 files, 10 modules, 5,253 LOC,
116 exports, 0 circular, 0 unused); `docs/architecture` stat/version markers
synced; CHANGELOG `[0.3.2]`; tagged `v0.3.2`.

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

## Done — v0.4.x feature line (v0.4.0–v0.4.5, COMPLETE 2026-06-04)

> **Sequencing (2026-06-04):** worked the v0.4.x set one sub-project
> at a time, foundation/safe-first. **Done:** HJB/costate co-training
> rewire (v0.4.0); README + linked-docs sweep (v0.4.1 docs); dead `LossConfig`
> fields removed (v0.4.1); KKT line-search Newton (v0.4.2); higher-fidelity
> nonlinear plants (v0.4.3); ParallelInferenceEngine hardening (v0.4.4); H∞
> adversary core (v0.4.5). **The v0.4.x line is COMPLETE.** Only follow-on left:
> the H∞ neural adversarial min-max training loop → **v0.5.0** (a new feature
> line, not part of v0.4.x).

- [x] **Dead `LossConfig` fields → wire-or-remove** (**DONE v0.4.1**): decided
  **remove** all 6 (`lambda_adjoint`, `alpha_attn`, `alpha_smooth`, `mu_lyap`,
  `beta_param`, `lambda_delta_u`) — they had zero usages (YAGNI; wiring is
  feature work and the sub-loss classes already carry their own weights).
  Behavior-preserving; YAML-backward-compatible. If a specific sub-loss is later
  wanted in the active loop, that's a focused feature (the classes exist).
- [x] **H∞ disturbance/adversary head — CORE (gap G1, Connection 7)** (**DONE
  v0.4.5**, brainstormed → spec → TDD). Built the *analytic* core: `solve_gare`
  (Hamiltonian–Schur GARE solver in `utils/lyapunov.py`; D defaults to B,
  fixed-γ with feasibility raise, γ→∞ recovers CARE) + `AdversaryHead`
  (`models/critic.py`; `w*=γ⁻²DᵀPe`, by construction from the critic gradient).
  Math reviewed clean. **Deferred to v0.5.0:** the neural adversarial **min-max
  training loop** (learned adversary network + worst-case co-training) — a new
  feature line on top of this analytic core.
- [x] **KKT projection robustness — line-search Newton** (**DONE v0.4.2**).
  Added a backtracking line search on the L∞ residual (`use_line_search=True`
  default + `line_search_max_halvings`): the iterate is now non-increasing and
  stays **bounded** on stiff constraints where the undamped full step diverged
  (atan(8·y)=0 case: ~1e7 → O(1)). Honest scope: line search *bounds* rather than
  fully *converges* the pathological gradient-vanishing case (atan's constraint
  gradient vanishes far from the root — not an overshoot problem; full Newton with
  curvature or a trust region would be needed to converge that, and the realistic
  smooth constraints already converge). Also fixed an off-by-one so
  `last_residual` reflects the returned iterate. IFT one-step unchanged.
- [x] **Cotrain HJB/costate critic-coupling — ADR + rewire** (discovered v0.3.3;
  **DONE in v0.4.0**). Resolved: HJB → opt-in critic regularizer applied via the
  critic optimizer (default off). Costate → **removed**, because brainstorming
  found `l_costate ≡ 0` (`λ̂ ≡ ∇V̂` by construction; the original "nonzero in
  general" assumption was wrong for the costate half — only HJB was a real
  signal). Identity 2 holds by construction. See the v0.4.0 release note above.
- [x] **Complete `ParallelInferenceEngine`** (**DONE v0.4.4**) — replaced the
  no-op adaptation with a real double-buffered IRL critic update (control thread
  feeds an `(e,u)` window; `_adaptation_update` deepcopies under the lock, takes
  one IRL Adam step off-lock, atomically swaps both critic + costate head) and
  added thread-exception capture (`error`/`check()`, fail-fast). Reviewed for
  thread-safety (no deadlock, no concurrent-deepcopy race). Remaining scaffold
  (documented): fixed `x_p`/`r`, cooperative scheduler, CBF `P` fixed at setup.
- [x] **Higher-fidelity example plants** (**DONE v0.4.3**) — `examples/plants.py`:
  `pendulum_step` (sin-gravity manipulator joint), `lateral_tyre_step` (tanh
  tyre-force-saturation lateral model), `rc_thermal_step` (2-node RC network +
  saturated heater). Each linearizes to the example's original linear surrogate
  so the LQR/CBF controller stays stable; all 3 closed loops verified bounded.
  Honest caveat: still illustrative, not hardware-validated.

## Carried-forward gaps / watch (discovered during the v0.3.2 sprint)

Grounded deferrals — none breaks tests today; each is verified against the
current source. Candidates for v0.4.0 or a later hardening pass.

1. **[RESOLVED v0.4.2] KKT projection robustness.** Added a backtracking line
   search to `KKTProjectionLayer.forward` (the robustness half of debt #2): the
   iterate is non-increasing and stays bounded on stiff constraints where the
   undamped full step diverged. (Honest caveat: bounds rather than fully converges
   the gradient-vanishing pathological case — a trust region / curvature Newton
   would be the next lever if ever needed; the realistic smooth constraints
   already converge.)
2. **[RESOLVED v0.3.3] Positivity regularizer was structurally inert in
   cotraining.** Investigating this gap revealed it was *not* a weight-tuning
   issue but a **wiring bug**: the `1e-3 * positivity` term lived in `l_total`
   (the PITNN objective), but it depends only on the critic's `W_c`, so
   `optimizer_pitnn.step()` never applied its gradient and the critic block's
   `zero_grad` wiped it. Fixed by applying positivity through the *critic*
   optimizer (guarded on a strictly-positive loss so it stays a no-op while P is
   PD and doesn't bias the IRL update's Adam schedule). New isolation test
   `test_cotrain_positivity_regularizer_repairs_indefinite_critic`. The same bug
   affects `l_hjb` / `l_costate` — bumped to v0.4.0 (see above) because fixing
   those is behavior-changing, unlike positivity.
3. **[RESOLVED v0.3.3] `_triu_pairs` cache hygiene.** `src/pits_mras/utils/
   lyapunov.py` now canonicalizes the device key (`_canonical_device_key`
   collapses `"cuda"` → `"cuda:<idx>"` when CUDA is available) and bounds the
   cache (`maxsize=128`). CPU behavior identical; tests
   `test_triu_pairs_canonicalizes_equivalent_devices_and_bounds_cache` +
   `test_canonical_device_key_cpu`.
4. **Example-test framework warmup is unavoidable at the test level** (watch).
   The ~15 s first-run cost in the example tests is torch higher-order-op /
   functorch lazy-init (see resolved #6), amortized across the suite. If CI
   wall-clock ever matters, the only real lever is framework-level (e.g. a
   session-scoped warmup fixture or splitting the autograd-heavy tests), not
   example-level — deliberately left alone for now.

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
