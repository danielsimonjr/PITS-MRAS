# PITS-MRAS — ROADMAP

> **Goal:** Build out the PITS-MRAS repository from its current scaffold-only state into a working, tested, documented Python framework that merges Physics-Informed Neural Networks, time-series deep learning, and Model-Reference Adaptive Systems, formally unified with RL / optimal control. [IP PAGE 1]
>
> This roadmap **operationalizes** [`ARCHITECTURE.md`](ARCHITECTURE.md) (the architectural source of truth — canonical module layout in §2, identity→module mapping in §3) and the two design PDFs in `docs/`:
> - *PITS-MRAS — A Mathematical and Architectural Blueprint* (the **Blueprint**, 12 pp).
> - *PITS-MRAS — Complete Implementation Plan for Claude Code* (the **Implementation Plan**, 48 pp, §0–§14).
>
> Page citations point at the parsed source `.txt` pages. `[IP PAGE N]` = Implementation Plan; `[BP PAGE N]` = Blueprint. Section numbers (`§N`) are verbatim from the Implementation Plan. The plan's own phase numbering (**Phase 1–9**, §4–§12) is reproduced exactly.
>
> *Items marked **(synthesis)** are organizational groupings inferred by this roadmap (e.g. milestone bundles), not verbatim from the sources. Everything else is grounded in the cited source page. Where the Blueprint and Implementation Plan diverge, the Implementation Plan's file/class names govern (it is the buildable artifact) and the divergence is flagged in §7.*

---

## 1. Current Status

**Verified ground truth (repo `pits-mras`, branch `master`, working tree clean):** scaffold-only. Tracked files are documentation + project metadata only — no `src/pits_mras/` package code, no tests, no examples:

- `setup.py`, `requirements.txt`, `.gitignore`, `LICENSE`
- Onboarding docs: `README.md`, `CONTRIBUTING.md`, `GITHUB_SETUP.md`, `PROJECT_SETUP_SUMMARY.md`
- Placeholder READMEs only: `src/README.md`, `tests/README.md`, `examples/README.md`
- `docs/` design corpus (ARCHITECTURE.md + the two design PDFs + the technical-spec `.md`/`.html`, PRD, project spec, validation report)

This matches the plan's §0 "Ground Truth": "src/README.md, tests/README.md, examples/README.md — all three implementation directories are empty stubs. The only substantive content is in docs/. Every .py file in this plan is net-new." [IP PAGE 1]

**Target end-state:** the full module tree of ARCHITECTURE.md §2 implemented; all `tests/` from §11 passing; CI green on Python 3.9/3.10/3.11 (flake8 + mypy + pytest); `pytest --cov=pits_mras` reporting **≥60% coverage**; all three examples runnable and generating plots. [IP PAGE 47, §13]

---

## 2. Guiding Priorities

The Blueprint's TL;DR is explicit about the highest-leverage-first ordering:

> "The single highest-leverage move is to recognize your MRAS Lyapunov function V(e)=eᵀPe is an LQR/tracking value function: replacing the manual V̇<0 → emergency backup heuristic with **(a) an Integral-RL policy-evaluation update (Vrabie–Lewis 2009) and (b) a CLF-CBF-QP safety filter (Ames et al. 2017)** gives you formal optimality + forward-invariance guarantees for near-zero implementation cost — these two are your best 'bang for buck.'" [BP PAGE 1, TL;DR]

The Blueprint formalizes this as **Tier 1 — do first (high rigor, low complexity)** [BP PAGE 9–10]:
1. **Integral-RL critic + synchronous policy iteration (Connection 1)** — critic head `V̂(e)`, integral-reinforcement accumulator, `L_IRL`, and the `K_fb ← R⁻¹BᵀP̂` policy-improvement step. [BP PAGE 9]
2. **CLF-CBF-QP safety filter (Connection 6)** — closed-form single-constraint projection; derive `h(e)=c−eᵀPe` so one `P` serves both CLF and CBF. [BP PAGE 9]
3. **Costate head with λ=∇V by construction (Connection 3)** — make the action head the autodiff gradient of a scalar value head. [BP PAGE 10]

**Sequencing reconciliation (synthesis):** the Implementation Plan mandates a strict in-order phased build — "Work through phases in order — each phase depends on the previous one" [IP PAGE 1] and "Execute phases in strict order" [IP PAGE 47]. The two priority capabilities therefore are **not** pulled ahead of their phases; instead they are the **critical-path, must-not-slip deliverables** living in:
- **Integral-RL critic** → `models/critic.py` + `losses/irl.py` (Phase 2 & 3), wired in `controllers/mras.py` + `training/cotrain.py` (Phase 4 & 5).
- **CLF-CBF-QP safety filter** → `controllers/safety.py` (Phase 4).
- **Costate head** → `models/critic.py` `CostateHead` (Phase 2).

Their foundations (Lyapunov/Riccati solvers in `utils/lyapunov.py`) are built in Phase 1, so the milestones below are sequenced to land Tier 1 as early as the dependency graph allows.

---

## 3. Phases

Reproduced **in order** from the Implementation Plan's §4–§12 (the phase-by-phase build, "Phase 1 Foundation" … "Phase 9 CI/CD") and cross-checked against ARCHITECTURE.md §2.1 "Subpackage Roles & Build Phases." All file paths match ARCHITECTURE.md §2 (assembled from the plan's inline `Create src/...` directives — see Gap G0 in §7). Acceptance gates are quoted from §13 "Implementation Order and Acceptance Criteria." [IP PAGE 47]

---

### Phase 1 — Foundation Layer
**Goal:** create the package scaffolding and pure-math utility modules — "No neural networks yet — only pure-math utilities that the network modules will import." [IP PAGE 6, §4]

- **Deliverables** [IP PAGES 6–15, §4.1–§4.5]:
  - `src/pits_mras/__init__.py` — exposes the eight top-level symbols (`PITNN`, `QuadraticCritic`, `MRASController`, `LinearReferenceModel`, `CLFCBFSafetyFilter`, `pretrain_pitnn`, `cotraining_loop`, `RealtimeInferenceEngine`), `__version__ = "0.1.0"`. [IP PAGE 6, §4.1]
  - Empty (docstring-only) `__init__.py` in `models/`, `controllers/`, `losses/`, `training/`, `inference/`, `utils/`. [IP PAGE 6, §4.1]
  - `src/pits_mras/config.py` — dataclass config: `NetworkConfig`, `PhysicsConfig`, `MRASConfig`, `SafetyConfig`, `LossConfig`, `TrainingConfig` → `PITSMRASConfig`, with `from_yaml`/`to_yaml`. [IP PAGES 7–10, §4.2]
  - `src/pits_mras/utils/lyapunov.py` — `solve_lyapunov`, `kleinman_iteration`, `solve_care`, `check_hurwitz`, `lyapunov_derivative`, `quadratic_basis` (scipy `solve_continuous_lyapunov`/`solve_continuous_are`). "the mathematical engine for all P-matrix computations." [IP PAGES 10–12, §4.3]
  - `src/pits_mras/utils/hamiltonian.py` — `make_skew_symmetric`, `make_positive_definite`, `port_hamiltonian_energy_loss`, `hamiltonian_positivity_loss`. [IP PAGES 13–14, §4.4]
  - `src/pits_mras/utils/pe_monitor.py` — `PEMonitor` (persistence-of-excitation Gram min-eigenvalue check + probing noise). [IP PAGES 14–15, §4.5]
  - (Cross-cutting, this phase per §2) Replace `requirements.txt` entirely + update `setup.py` to `name="pits_mras"`, `version="0.1.0"`, src-layout, `extras_require` dev/logging. [IP PAGES 2–3, §2] — **see conflict G2 in §7.**
- **Depends on:** nothing.
- **Acceptance criteria / gate** [IP PAGE 47, §13]: `python -c "from pits_mras.utils.lyapunov import solve_lyapunov; import numpy as np; A = -np.eye(2); Q = np.eye(2); P = solve_lyapunov(A, Q); print(P)"` must print `[[0.5, 0], [0, 0.5]]` (A=−I ⇒ P=½I satisfies −P+(−P)=−I).
- **Tasks:**
  - [ ] Decide and record the `setup.py`/`requirements.txt` overwrite (resolve G2) before scaffolding.
  - [ ] Create the package + six empty subpackage `__init__.py` files and the top-level `__init__.py` symbol exports.
  - [ ] Implement `config.py` (all six dataclasses + master `PITSMRASConfig` + `from_yaml`/`to_yaml`).
  - [ ] Implement `utils/lyapunov.py` (six functions; scipy-backed).
  - [ ] Implement `utils/hamiltonian.py` (four functions).
  - [ ] Implement `utils/pe_monitor.py` (`PEMonitor`).
  - [ ] Run the §13 Phase-1 sanity command; confirm it prints `[[0.5,0],[0,0.5]]`.

---

### Phase 2 — Neural Network Models
**Goal:** build the PITNN encoder, port-Hamiltonian decoder, and the new critic/costate heads. [IP PAGE 16, §5]

- **Deliverables** [IP PAGES 16–28, §5.1–§5.4]:
  - `src/pits_mras/models/attention.py` — `PhysicsInformedAttention` (temporal + physical + error-driven, learned 3-way gate; `attention_regularization_loss`). [IP PAGE 16, §5.1]
  - `src/pits_mras/models/decoders.py` — `HamiltonianNet`, `DissipationNet`, `PortHamiltonianDecoder` (implements §3.1: `f̂_θ = J∇H − R_θ q̇ + B u + W_corr c_t + b_corr`). [IP PAGE 18, §5.2]
  - `src/pits_mras/models/critic.py` — **`QuadraticCritic`** (`V̂=Wᵀφ(e)`; `forward`, `gradient`, `extract_P`, `positivity_loss`; optional `nonlinear_residual`) and **`CostateHead`** (`λ̂=∇V̂`, `u*=−R⁻¹Bᵀλ̂`). **NEW — Identity 1 & 2.** [IP PAGE 22, §5.3] — *priority modules (a) critic + (c) costate.*
  - `src/pits_mras/models/pitnn.py` — `PITNN` (embedding → causal LSTM → attention → port-Hamiltonian decoder; Algorithm 1). [IP PAGE 25, §5.4]
- **Depends on:** Phase 1 (`config.py`, `utils/hamiltonian.py`, `utils/lyapunov.quadratic_basis`).
- **Acceptance criteria / gate** [IP PAGE 47, §13]: `pytest tests/test_models.py -v` — all model unit tests pass (`test_dissipation_matrix_psd`, `test_J_skew_symmetric`, `test_hamiltonian_positive`). *(Note: `tests/test_models.py` is formally authored in Phase 8 §11.4; per §13 this model gate runs against those tests once written — see sequencing risk in §7.)*
- **Tasks:**
  - [ ] Implement `models/attention.py` (`PhysicsInformedAttention` + regularization loss).
  - [ ] Implement `models/decoders.py` (`HamiltonianNet`, `DissipationNet`, `PortHamiltonianDecoder`).
  - [ ] Implement `models/critic.py` `QuadraticCritic` (basis init near P≈I, `extract_P`, `positivity_loss`).
  - [ ] Implement `models/critic.py` `CostateHead` (gradient-of-critic action, Identity 2 by construction).
  - [ ] Implement `models/pitnn.py` `PITNN.forward` (Algorithm 1; causal/forward-only LSTM).
  - [ ] Verify against `test_models.py` (R_θ PSD, J skew-symmetric, H>0).

---

### Phase 3 — Loss Functions
**Goal:** all loss modules, including the new Integral-RL Bellman loss and HJB residual loss, plus the `TotalLoss` aggregator. [IP PAGE 28, §6]

- **Deliverables** [IP PAGES 28–31, §6.1–§6.6]:
  - `src/pits_mras/losses/physics.py` — `PhysicsLoss` (`λ₁L_energy+λ₂L_PDE+λ₃L_BC+λ₄L_sym`; ported from technical spec §2.2). [IP PAGE 28, §6.1]
  - `src/pits_mras/losses/temporal.py` — `MultiStepPredictionLoss`, `AttentionRegularizationLoss`, `TemporalSmoothnessLoss`, `TemporalLoss`. [IP PAGE 28, §6.2]
  - `src/pits_mras/losses/stability.py` — `LyapunovConstraintLoss`, `ParameterBoundednessLoss`, `ControlEffortLoss`, `MRASStabilityLoss`. [IP PAGE 29, §6.3]
  - `src/pits_mras/losses/irl.py` — **`IRLBellmanAccumulator`, `IRLBellmanLoss`. NEW — Identity 1.** [IP PAGE 29, §6.4] — *priority module (a).*
  - `src/pits_mras/losses/hjb.py` — **`HJBResidualLoss`, `LyapunovDecreaseEnforcer`. NEW — Identity 8.** [IP PAGE 31, §6.5]
  - `src/pits_mras/losses/__init__.py` — `TotalLoss` aggregator + per-sub-loss TensorBoard/wandb logging (`loss/physics`, `loss/temporal`, `loss/stability`, `loss/irl`, `loss/hjb`, `loss/costate`, `loss/data`). [IP PAGE 31, §6.6]
- **Depends on:** Phase 2 (critic/costate for IRL & HJB), Phase 1 (`utils/lyapunov.lyapunov_derivative`).
- **Acceptance criteria / gate** [IP PAGE 47, §13]: `pytest tests/test_irl.py tests/test_identity_costate.py -v` (`test_irl_bellman_error_zero_at_true_value`, `test_irl_loss_decreases_with_correct_update`, `test_costate_equals_grad_V`, `test_optimal_control_equals_lqr_gain`). *(Tests authored in Phase 8 — see §7 sequencing risk.)*
- **Tasks:**
  - [ ] Implement `losses/physics.py` `PhysicsLoss` (energy from decoder + PDE/BC/sym callables — formula details live in technical-spec `.md`, see G4).
  - [ ] Implement `losses/temporal.py` (four classes).
  - [ ] Implement `losses/stability.py` (four classes; import `lyapunov_derivative`).
  - [ ] Implement `losses/irl.py` `IRLBellmanAccumulator` (trapezoidal `∫r dτ`) + `IRLBellmanLoss` (`½E[δ_IRL²]`).
  - [ ] Implement `losses/hjb.py` `HJBResidualLoss` + `LyapunovDecreaseEnforcer`.
  - [ ] Implement `losses/__init__.py` `TotalLoss` aggregator with per-sub-loss logging.
  - [ ] Verify against `test_irl.py` (Bellman error ≈ 0 at true value) and `test_identity_costate.py`.

---

### Phase 4 — Controllers
**Goal:** reference model, the CLF-CBF safety filter, and the actor-critic MRAS controller. [IP PAGE 31, §7]

- **Deliverables** [IP PAGES 31–37, §7.1–§7.3]:
  - `src/pits_mras/controllers/reference_models.py` — `LinearReferenceModel` (verifies Hurwitz `A_m`; on init solves Lyapunov for `P` and runs `kleinman_iteration` → `P_opt, K_opt`; buffers `P, P_opt, K_opt, R_inv`; Euler `step`). [IP PAGE 31, §7.1]
  - `src/pits_mras/controllers/safety.py` — **`CLFCBFSafetyFilter`** (closed-form CBF projection; `h(e)=c−eᵀPe`, `L_f h=−2eᵀPA_m e`, `L_g h=−2eᵀPB`; `cbf_constraint_loss`). **NEW — Identity 3.** [IP PAGE 33, §7.2] — *priority module (b).*
  - `src/pits_mras/controllers/mras.py` — `MRASController` (critic + costate head + `K_ff` + `compensator` + CBF filter; `mras_regressor` φ_c=[eᵀ,rᵀ,x_pᵀ]ᵀ; LQR warm-start of critic to `P_opt`). [IP PAGE 35, §7.3]
- **Depends on:** Phase 2 (`critic.py`), Phase 3 (loss hooks), Phase 1 (`utils/lyapunov`).
- **Acceptance criteria / gate** [IP PAGE 47, §13]: `pytest tests/test_safety.py tests/test_identity_costate.py -v` (`test_cbf_projects_unsafe_control`, `test_cbf_identity_when_safe`, `test_cbf_forward_invariance` [100-step sim stays in safe set], plus the costate tests). *(Tests authored in Phase 8.)*
- **Tasks:**
  - [ ] Implement `controllers/reference_models.py` `LinearReferenceModel` (Hurwitz check, Lyapunov + Kleinman on init, Euler step).
  - [ ] Implement `controllers/safety.py` `CLFCBFSafetyFilter` (closed-form projection + `cbf_constraint_loss`).
  - [ ] Implement `controllers/mras.py` `MRASController` (assemble critic/costate/K_ff/compensator/CBF; LQR warm-start; `mras_regressor`).
  - [ ] Verify against `test_safety.py` (unsafe→projected, safe→identity, 100-step forward invariance).

---

### Phase 5 — Training Pipelines
**Goal:** the pre-training curriculum, the IRL-extended co-training loop (actor-critic), and the standalone offline IRL critic trainer. [IP PAGE 38, §8]

- **Deliverables** [IP PAGES 38–40, §8.1–§8.3]:
  - `src/pits_mras/training/pretrain.py` — `pretrain_pitnn`; 3-stage curriculum (1A physics-only epochs 1–1000; 1B cosine-anneal `λ_data` 1001–3000; 1C add `L_temporal` warm-up 3001–5000); validation criterion `L_physics<ε_tol`. Algorithm 2. [IP PAGE 38, §8.1]
  - `src/pits_mras/training/cotrain.py` — `cotraining_loop`; "the most critical training file." Extends Algorithm 3 with: IRL critic update (separate `critic_optimizer` Adam lr=1e-3, grad-clip 1.0) + policy improvement `K_new=R⁻¹BᵀP̂`; HJB update (if `λ_hjb>0`); costate consistency `L_costate`; critic positivity `1e-3·L_pos`; CBF constraint `0.1·L_cbf`. Two optimizers (`optimizer_pitnn` Adam lr=1e-4). [IP PAGES 39–40, §8.2]
  - `src/pits_mras/training/irl_trainer.py` — offline batch least-squares critic pre-training; stops when `‖P̂−P_opt‖_F/‖P_opt‖_F < 0.01`. [IP PAGE 40, §8.3]
- **Depends on:** Phase 2, Phase 3, Phase 4 (the full model+loss+controller stack).
- **Acceptance criteria / gate** [IP PAGE 47, §13]: `pytest tests/test_smoke.py -v` — smoke tests pass with **no NaN losses** (`test_pretrain_one_epoch`, `test_cotrain_one_episode` produce finite scalars). *(Tests authored in Phase 8.)*
- **Tasks:**
  - [ ] Implement `training/pretrain.py` `pretrain_pitnn` (stages 1A/1B/1C schedules + `λ_data` 0.5-reduction safeguard).
  - [ ] Implement `training/cotrain.py` `cotraining_loop` base + the five IRL/HJB/costate/positivity/CBF additions (see G5: base Algorithm 3 body is prose-only).
  - [ ] Wire the two optimizers (`optimizer_pitnn` lr=1e-4, `critic_optimizer` lr=1e-3).
  - [ ] Implement `training/irl_trainer.py` offline batch least-squares (stop on relative P error < 0.01).
  - [ ] Verify against `test_smoke.py` (one pretrain epoch + one cotrain episode, all losses finite).

---

### Phase 6 — Inference Engine
**Goal:** real-time closed-loop inference with the CBF filter, plus the parallel thread architecture. [IP PAGE 40, §9]

- **Deliverables** [IP PAGES 40–43, §9.1–§9.2]:
  - `src/pits_mras/inference/realtime.py` — `RealtimeInferenceEngine.step(x_p, r, dt)`: measure → bounded deque buffers → PITNN forward → reference-model step + error → controller forward → **CBF safety filter (replaces heuristic V̇<0 check)** → apply `u_safe` → log `V̂, h_CBF, ‖e‖`. `@torch.no_grad()` + `threading.Lock`. [IP PAGE 40, §9.1]
  - `src/pits_mras/inference/parallel.py` — `ControlThread` (1 kHz), `AdaptationThread` (100 Hz; IRL update + policy improvement; `copy.deepcopy` + atomic double-buffer swap), `MonitorThread` (10 Hz); graceful shutdown via `threading.Event`. [IP PAGE 42, §9.2]
- **Depends on:** Phase 2 (`pitnn.py`), Phase 4 (`mras.py`, `reference_models.py`), Phase 5 (adaptation logic reused by `AdaptationThread`).
- **Acceptance criteria / gate** [IP PAGE 47, §13]: Run `examples/robotic_manipulator.py` for 100 steps; plots are generated without error. (Also covered by `test_smoke.test_full_forward_pass_no_crash` — "run 10 steps of RealtimeInferenceEngine, no exceptions.") [IP PAGE 45, §11.6]
- **Tasks:**
  - [ ] Implement `inference/realtime.py` `RealtimeInferenceEngine` (deque buffers, lock, closed-loop step, logging).
  - [ ] Implement `inference/parallel.py` three threads (1 kHz / 100 Hz / 10 Hz) + double-buffer critic swap + `threading.Event` shutdown.
  - [ ] Verify the 100-step robotic-manipulator run produces plots without error.

---

### Phase 7 — Examples
**Goal:** three runnable closed-loop demonstrations with the mandated diagnostic plots. [IP PAGE 43, §10]

- **Deliverables** [IP PAGE 43, §10.1–§10.3]:
  - `examples/robotic_manipulator.py` — 2-DOF planar manipulator; H = ½q̇ᵀM(q)q̇+V(q); sinusoidal joint-angle reference. Plots: (a) ‖e(t)‖, (b) V̂(e(t)) (monotone decrease after warm-up), (c) CBF activation flag, (d) `‖P̂−P_CARE‖_F/‖P_CARE‖_F` critic-convergence. [IP PAGE 43, §10.1]
  - `examples/autonomous_vehicle.py` — lateral control at 80 km/h; wind-gust disturbance `Δ(t)=0.5 sin(2πt/10)`; with-CBF vs without-CBF lane-departure comparison. [IP PAGE 43, §10.2]
  - `examples/building_hvac.py` — thermal-zone control; thermal-energy Hamiltonian; energy savings vs PID baseline + seasonal P̂ adaptation. [IP PAGE 43, §10.3]
- **Depends on:** Phase 6 (inference engine), Phase 5 (trained models).
- **Acceptance criteria / gate:** the Phase-6 §13 gate (robotic-manipulator 100-step run generates plots without error) is the only explicitly stated example gate; **per-example numerical acceptance not specified in source** for autonomous_vehicle / building_hvac beyond the described plots. [IP PAGE 47, §13]
- **Tasks:**
  - [ ] Implement `examples/robotic_manipulator.py` (4-panel diagnostic plot).
  - [ ] Implement `examples/autonomous_vehicle.py` (with-/without-CBF comparison under wind gust).
  - [ ] Implement `examples/building_hvac.py` (energy savings vs PID + seasonal adaptation).

---

### Phase 8 — Tests
**Goal:** the full test suite covering each identity, safety, models, IRL, and end-to-end smoke. [IP PAGE 43, §11]

- **Deliverables** (verbatim test names) [IP PAGES 43–45, §11.1–§11.6]:
  - `tests/test_identity_lyapunov_value.py` — `test_kleinman_converges_to_care`, `test_irl_critic_converges_to_lyapunov_P`, `test_quadratic_basis_reconstructs_P` (Identity 1). [IP PAGE 43, §11.1]
  - `tests/test_identity_costate.py` — `test_costate_equals_grad_V`, `test_optimal_control_equals_lqr_gain` (Identity 2). [IP PAGE 44, §11.2]
  - `tests/test_safety.py` — `test_cbf_projects_unsafe_control`, `test_cbf_identity_when_safe`, `test_cbf_forward_invariance` (Identity 3 / forward invariance). [IP PAGE 44, §11.3]
  - `tests/test_models.py` — `test_dissipation_matrix_psd`, `test_J_skew_symmetric`, `test_hamiltonian_positive`. [IP PAGE 44, §11.4]
  - `tests/test_irl.py` — `test_irl_bellman_error_zero_at_true_value`, `test_irl_loss_decreases_with_correct_update`. [IP PAGE 45, §11.5]
  - `tests/test_smoke.py` — `test_full_forward_pass_no_crash`, `test_pretrain_one_epoch`, `test_cotrain_one_episode`. [IP PAGE 45, §11.6]
- **Depends on:** Phases 1–6 (each test targets a module from a prior phase). **NOTE (synthesis):** the §13 acceptance gates for Phases 2–5 *invoke these test files*, so in practice the relevant test files must be authored alongside their target phase even though §11 nominally calls this "Phase 8" — see the sequencing risk in §7.
- **Acceptance criteria / gate** [IP PAGE 47, §13]: all tests pass; `pytest --cov=pits_mras` reports **≥60% coverage** (the Phase 7–9 combined gate).
- **Tasks:**
  - [ ] Author `test_identity_lyapunov_value.py` (3 tests, Identity 1).
  - [ ] Author `test_identity_costate.py` (2 tests, Identity 2).
  - [ ] Author `test_safety.py` (3 tests, Identity 3 / forward invariance).
  - [ ] Author `test_models.py` (3 tests, port-Hamiltonian structure).
  - [ ] Author `test_irl.py` (2 tests, IRL loss).
  - [ ] Author `test_smoke.py` (3 end-to-end tests, no NaN).
  - [ ] Confirm `pytest --cov=pits_mras` ≥ 60%.

---

### Phase 9 — CI/CD
**Goal:** continuous integration + modern tooling config. [IP PAGE 45, §12]

- **Deliverables** [IP PAGES 45–47, §12]:
  - `.github/workflows/ci.yml` — matrix Python 3.9/3.10/3.11; steps: `pip install -e ".[dev]"`, flake8 (`--max-line-length=100 --ignore=E203,W503`), mypy (`--ignore-missing-imports`), `pytest --cov=pits_mras --cov-report=xml`, codecov upload. [IP PAGE 46, §12]
  - `pyproject.toml` — `[tool.black]` (line-length 100, py39/310/311), `[tool.isort]` (profile black), `[tool.mypy]`, `[tool.pytest.ini_options]`. [IP PAGES 46–47, §12]
- **Depends on:** Phase 8 (tests exist to run in CI).
- **Acceptance criteria / gate** [IP PAGE 47, §13]: "Phase 7–9: All tests pass in CI; `pytest --cov=pits_mras` reports ≥60% coverage."
- **Tasks:**
  - [ ] Create `.github/workflows/ci.yml` (checkout → setup-python matrix → install → flake8 → mypy → pytest+cov → codecov).
  - [ ] Create `pyproject.toml` (black/isort/mypy/pytest config).
  - [ ] Confirm CI is green on all three Python versions with ≥60% coverage.

---

## 4. Milestones *(synthesis — groupings inferred; all phase content is from the source)*

Grouping bundles the plan's nine phases into four delivery milestones. Exit criteria are the conjunction of the member phases' §13 gates. The Blueprint Tier-1 priorities (a/b/c) are annotated where they land.

| Milestone | Phases | Theme | Exit criterion |
|---|---|---|---|
| **M1 — Core scaffold + reference model foundations** | 1 | Importable package, config, Lyapunov/Riccati engine, PE monitor | §13 Phase-1 sanity command prints `[[0.5,0],[0,0.5]]` [IP PAGE 47] |
| **M2 — RL-unification: models + losses (critic, costate, IRL)** | 2, 3 | PITNN + port-Hamiltonian decoder; **critic (a)** + **costate head (c)**; physics/temporal/stability + **IRL (a)** + HJB losses | `test_models.py` passes [IP PAGE 47]; `test_irl.py` + `test_identity_costate.py` pass (Bellman error ≈ 0; costate = ∇V) [IP PAGE 47] |
| **M3 — Safety + robustness + training (CLF-CBF-QP, actor-critic)** | 4, 5 | Reference model, **CLF-CBF-QP filter (b)**, actor-critic `MRASController`; pretrain + IRL co-training + offline IRL trainer | `test_safety.py` passes incl. 100-step forward invariance [IP PAGE 47]; `test_smoke.py` passes with no NaN losses [IP PAGE 47] |
| **M4 — Inference, examples, tests & CI (end-to-end validated)** | 6, 7, 8, 9 | Real-time + parallel inference; three examples; full test suite; CI matrix | Robotic-manipulator 100-step run plots without error [IP PAGE 47]; all tests pass in CI; `pytest --cov=pits_mras` ≥ 60% [IP PAGE 47] |

**Project-level done (all milestones):** all nine phases complete; all §11 tests green in CI on py3.9/3.10/3.11; coverage ≥60%; three examples runnable. [IP PAGE 47, §13]

---

## 5. Cross-Cutting Tracks *(run alongside the phases)*

- **Configuration (§4.2):**
  - [ ] All hyperparameters centralized in `config.py` dataclasses; `from_yaml`/`to_yaml`; defaults per §4.2 (`input_dim=10`, `hidden_dim=128`, `output_dim=4`, `memory_horizon=50`, `state_dim=4`, `control_dim=2`, loss weights `λ_irl=1.0`, `λ_hjb=0.01`, …). [IP PAGES 7–10]
- **Dependencies (§2):**
  - [ ] `requirements.txt` replaced entirely; `setup.py` updated (`pits_mras` / `0.1.0` / src-layout). [IP PAGES 2–3] — **resolve conflict G2 first (§7).**
  - [ ] `cvxpy` kept commented-out (only needed for >1 CBF constraint). [IP PAGE 3]
  - [ ] Review whether `control>=0.9.4` is actually used (code uses scipy — gap G3, §7). [IP PAGE 2 vs PAGES 10–12]
- **Testing (§11) — runs continuously, not just Phase 8:**
  - [ ] Author each test file alongside the phase whose §13 gate invokes it (see §7 sequencing risk).
  - [ ] Numerical/structural checks: Bellman-error-zero, costate=∇V, CBF forward invariance (100-step sim), R_θ PSD, J skew-symmetric, H>0, no-NaN smoke. [IP PAGES 43–45]
  - [ ] Target `pytest --cov=pits_mras` ≥ 60%. [IP PAGE 47]
- **CI / tooling (§12):**
  - [ ] flake8 (max-line 100, ignore E203/W503) + mypy (ignore-missing-imports) + pytest+coverage→codecov, py3.9/3.10/3.11. [IP PAGES 45–47]
- **Documentation:**
  - [ ] Keep `docs/ARCHITECTURE.md` as the authoritative layout/identity map; update example/result docs as Phase 7 lands.
  - [ ] Honor §14 conventions in all code: tensor shapes `[batch, dim]`/`[batch, T, dim]` (never `[dim, batch]`); device from `cfg.training.device` (never hardcode `torch.cuda`); `e.requires_grad_(True)` before critic for costate, `create_graph=True` for adjoint loss; normalize critic inputs. [IP PAGES 47–48, §14]
- **Numerical-stability invariants (§14):**
  - [ ] `A_m` Hurwitz (verified in `LinearReferenceModel`); `P>0`, `R_θ=LᵀL`, `J=−Jᵀ`, `H>0` (softplus); causal forward-only LSTM. [IP PAGES 47–48]
  - [ ] Persistence-of-excitation: use `PEMonitor` for IRL/ADP convergence; probing noise biases estimates unless handled (Blueprint caveat). [IP PAGE 14; BP PAGE 9]

---

## 6. Open Questions / Risks

**Sequencing constraints & pitfalls (from the plan + Blueprint caveats):**
- **Strict in-order build is mandatory.** "Work through phases in order — each phase depends on the previous one." [IP PAGE 1] "Execute phases in strict order. Each phase has a verifiable acceptance criterion before proceeding." [IP PAGE 47]
- **The §13 acceptance gates for Phases 2–6 invoke `tests/test_*.py` files that §11 nominally assigns to "Phase 8."** This is an internal sequencing tension in the source: you cannot satisfy the Phase-2 gate (`pytest tests/test_models.py`) without having written `test_models.py`. **Resolution (synthesis): author each test file alongside its target phase**, treating §11 as the catalog and §13 as the true ordering. Flagged, not silently reordered.
- **Math identities in §3 are "load-bearing; implement them incorrectly and the theoretical guarantees collapse." Do not approximate or simplify them.** [IP PAGE 1, §3]
- **Persistence-of-excitation bias:** every IRL/ADP method needs PE; "injecting probing noise biases estimates unless handled." [BP PAGE 9]
- **HJB loss is not a guaranteed win:** start `λ_HJB=0.01`, treat as a tunable regularizer (HJBPPO found no consistent improvement). [IP PAGE 5; BP PAGE 7]
- **Factor-of-½ convention:** keep `V=eᵀPe` vs `½xᵀPx` bookkeeping consistent when wiring `R⁻¹BᵀP`. [BP PAGE 10]
- **DPG compatibility:** the deterministic-policy-gradient identity is unbiased only with a compatible critic. [BP PAGE 10]

**Conflicts flagged between `docs/ARCHITECTURE.md` and `implplan.txt`:** ARCHITECTURE.md does not contradict the implplan — it *documents* the implplan's own internal gaps/ambiguities as G0–G9. The load-bearing ones for sequencing:
- **G0 — no single verbatim "§2 target tree" exists in the plan.** The plan's §2 is "Updated Dependencies"; file paths are scattered inline as `Create src/...` across §4–§12. ARCHITECTURE.md §2 assembled the canonical tree from those directives. This ROADMAP's paths come from that assembled tree. [IP PAGE 2; ARCHITECTURE.md §2]
- **G1 — Blueprint vs Plan disagree on the "three new heads."** Blueprint says critic / **disturbance-adversary** / CBF [BP PAGE 1]; the Plan concretely builds critic / **costate** / CBF and gives **no** module for the H∞ adversary head (nor for SAC/entropy Connection 5 or TD-MPC2 Connection 9). Per the precedence rule, the **Plan's heads govern**; the adversary/H∞ module is *not specified for build*. [BP PAGE 1,6 vs IP §5,§7]
- **G2 — `setup.py`/`requirements.txt` conflict with the existing scaffold:** current scaffold is `name="pits-mras"`, `version="1.0.0"`, no `control`/`isort`; the plan mandates `name="pits_mras"`, `version="0.1.0"`, adds `control`/`torchvision`/`pytest-cov`/`isort`, and says "Replace requirements.txt entirely." **Decision required in Phase 1.** [IP PAGES 2–3]
- **G3 — `control>=0.9.4` may be unused:** listed as a dep ("Riccati, Lyapunov solvers") but the actual `lyapunov.py` uses scipy. [IP PAGE 2 vs PAGES 10–12]

**Items marked unspecified / inferred:**
- **Milestone groupings (M1–M4) are synthesis** — the source defines nine phases, not four milestones.
- **Tolerances are not numerically specified** for "within tolerance" gates (IRL→P convergence, costate equality, etc.) beyond the explicit `‖P̂−P_opt‖_F/‖P_opt‖_F < 0.01` (Phase 5 IRL trainer) and the `[[0.5,0],[0,0.5]]` Phase-1 sanity value. **Choose per-test and document.**
- **G4 — the technical-spec `docs/PITS-MRAS — Physics-Informed...md` (1,543 lines) is the plan's "source of truth for the existing design"** and holds the exact `L_physics` PDE operator, `temporal.py`, and `stability.py` formulas; ARCHITECTURE.md notes that file was not available to its author. Consult it when implementing those Phase-3 losses. [IP PAGE 1, §0]
- **G5 — `cotrain.py` base loop (Algorithm 3) is prose-only;** only the IRL/HJB/costate/positivity/CBF *additions* are given as code ("line 52 of the existing algorithm"). The base body must be derived from the technical spec. [IP PAGE 39]
- **G6 — `pretrain_pitnn` / `cotraining_loop` full signatures/return types are not given** (only stage logic). [IP PAGES 6, 38–40]
- **G7 — dataset / trajectory data source is unspecified** (format/generation/loader); there is **no `data/` module** in the layout. Pretraining and the offline IRL trainer assume "trajectory data" / "fixed dataset of trajectories" without defining it. [IP PAGES 38, 40]
- **G8 — MIMO control input is incomplete in the decoder** (`f_ctrl = B_val * u.sum(...)` annotated "simplified; generalize for MIMO"). [IP PAGE 21]
- **Per-example numerical acceptance** for `autonomous_vehicle.py` and `building_hvac.py` is **gate not specified in source** beyond the described plots; only the robotic-manipulator 100-step plot-generation gate is explicit. [IP PAGE 47]

---

*This ROADMAP is the operational companion to `docs/ARCHITECTURE.md`. If any file path or identity label here ever diverges from ARCHITECTURE.md §2 / §3, ARCHITECTURE.md is authoritative and this file must be corrected.*
