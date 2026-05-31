# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

- **Status:** Phases 1–6 (foundation, models, losses, controllers, training,
  inference) are implemented and tested. Phases 7–9 (examples, full test suite /
  coverage, CI finalization) remain. Implementation proceeds per `docs/ROADMAP.md`.
- Verified gates after Phase 6: `flake8 src tests` → 0; `mypy src` → 0;
  `pytest` → 129 passed, 1 skipped (the Phase-2 IRL-convergence placeholder);
  `import pits_mras` → 0.1.0.
- **CI install:** still `pip install -e . --no-deps` plus the dev toolchain in the
  workflow. Phase 1 utils import numpy/scipy/torch, so CI now also installs the
  Phase-1 runtime deps (CPU-only torch) before running the gates.
- **Python floor:** the Implementation Plan stated a 3.9 baseline, but the current
  mypy release dropped `python_version = 3.9` support and `torch>=2.0.0` requires
  3.10+, so the CI matrix and tool configs use **3.10/3.11/3.12**.
