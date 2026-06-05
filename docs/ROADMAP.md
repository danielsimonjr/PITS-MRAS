# PITS-MRAS — ROADMAP

> **Forward-looking roadmap.** The v0.1.0 → v0.4.5 build is **complete and released**
> (Foundation, models, losses, controllers, training, inference, examples, tests, CI,
> plus the PCML layer and the analytic H∞ core). The shipped history of record lives in
> [`../CHANGELOG.md`](../CHANGELOG.md); the implemented design is documented under
> [`architecture/`](architecture/) (`ARCHITECTURE.md` is the canonical layout/identity map,
> with `OVERVIEW.md`, `COMPONENTS.md`, `DATAFLOW.md`, `API.md`, `DEPENDENCY_GRAPH.md`).
>
> This file tracks **proposed and not-yet-implemented work only**. The completed nine-phase
> build plan was removed on 2026-06-05 (it had gone fully stale — CHANGELOG + `architecture/`
> are the durable record).

---

## 1. Status

**v0.4.5 — framework complete.** Gates green: pytest 211/0, flake8 + mypy clean, dependency
graph 0 circular / 0 unused. The examples run; performance figures in the docs are design
targets on illustrative nonlinear plants, **not** hardware-validated.

---

## 2. Proposed Improvements

Derived 2026-06-05 from a four-agent deep-research pass (one agent inventoried the codebase;
three surveyed the SOTA in physics-informed, physics-constrained, time-series, and
adaptive/robust-control ML) and cross-checked against the actual source. Each item is
anchored to a real file and a representative primary source. **Effort:** S (hours), M (days),
L (a feature line with its own brainstorm → spec → TDD). Nothing here is scheduled yet — these
are candidates, grouped by the three improvement axes.

> Proposals that the research surfaced but that are **already implemented** were excluded; see
> §4 for that honesty list. Each proposal below was verified absent from the current source.

### 2.1 Capabilities

1. **H∞ neural adversarial min-max training loop.**
   Replace the closed-form `AdversaryHead` (`src/pits_mras/models/critic.py:183–215`,
   `w* = γ⁻²DᵀPe`) with a *learned* disturbance network and co-train a
   protagonist/critic/adversary triple on the HJI equation. Warm-start from the analytic GARE
   (`src/pits_mras/utils/lyapunov.py` `solve_gare`) and use it as the correctness oracle
   (γ → ∞ must reproduce the existing CARE behaviour — a free regression test). Stabilize the
   min-max with two-timescale gradient descent-ascent + a Stackelberg (leader–follower)
   framing rather than naive simultaneous updates.
   *Sources:* three-network actor–critic–disturbance ADP (Neurocomputing 2016; Soft Computing
   2023); MAGICS, locally-convergent neural min-max for safe control (arXiv:2409.13867);
   RRL-Stack (arXiv:2202.09514); off-policy H∞ RL (Modares–Lewis, IEEE TNNLS 2015).
   *Effort:* **L.** (This is the previously-flagged G1 remainder — see §3.)

2. **Deep-Koopman lifting head.**
   Sequence modeling today is LSTM-only (`src/pits_mras/models/pitnn.py:63–69`); there is no
   state-space/Koopman path. Learn an encoder to a lifted latent where dynamics are
   approximately linear `(A,B)`, then run the **existing** quadratic critic / `solve_gare` /
   CLF-CBF stack on the lifted coordinates — bridging the nonlinear plants to the verifiable
   linear core. (Alternative if upgrading the backbone instead of lifting: an S5/Mamba
   state-space layer, cheaper than attention at long horizons.)
   *Sources:* Lusch et al., Nature Comms 2018; Korda & Mezić, Automatica 2018; Mamba
   (arXiv:2312.00752). *Effort:* **M–L.**

3. **Uncertainty quantification: deep ensembles + conformal calibration.**
   No UQ exists anywhere in the source. Train K seeds of the PITNN (epistemic UQ) and wrap
   predictions in adaptive/copula conformal intervals for finite-sample coverage on the time
   series. *Caveat:* exact coverage degrades under non-exchangeability; adaptive variants
   restore asymptotic coverage only.
   *Sources:* Lakshminarayanan et al., NeurIPS 2017; Gibbs & Candès, NeurIPS 2021; copula-CP
   (arXiv:2212.03281). *Effort:* **S–M.**

4. **Long-horizon rollout-stability + conservation-drift diagnostics.**
   The port-Hamiltonian decoder makes the energy residual vanish by construction
   (`src/pits_mras/models/decoders.py:199–202`) but this is never *validated* over a rollout,
   and no stability metrics exist. Add diagnostics for Hamiltonian/energy drift over
   autoregressive rollout, Valid-Prediction-Time, and the rollout-Jacobian spectral radius
   (error-amplification rate). Turns structural physics guarantees into measured ones.
   *Sources:* standard conservation-drift / VPT / Jacobian-growth rollout metrics (the
   specific 2026 preprints are pre-peer-review; the metrics themselves are established).
   *Effort:* **S.**

### 2.2 Efficiencies

5. **Vectorize the KKT constraint-Jacobian (and ∇V̂ / HJB) with `torch.func`.**
   `src/pits_mras/models/pcml.py:225–231` builds the constraint Jacobian with a per-constraint
   Python `for` loop of `torch.autograd.grad(..., retain_graph=True)`; no `torch.func`/`vmap`
   is used anywhere. Replace with batched `jacrev`/`vmap` — faster than per-row autograd and
   clearer. *Pitfall:* requires the layer in functional (`functional_call`) form.
   *Source:* PyTorch jacobians/hessians functional-transform guidance. *Effort:* **M.**

6. **Differentiable GARE / CARE via implicit differentiation.**
   `utils/lyapunov.py` solves via SciPy on float64 numpy — correct, but graph-breaking (the
   gain is a constant w.r.t. NN params). It is init-only today, so this is **an enabler for
   proposal #1**, not a standalone speedup: a custom `torch.autograd.Function` that solves the
   forward under `no_grad` and the backward via the linearized Lyapunov/Sylvester sensitivity
   (no solver unrolling). *Sources:* `dare-torch`; arXiv:2011.11430. *Effort:* **M.**

7. **Reference-swap critic hot-swap in the parallel engine.** — **EVALUATED 2026-06-05:
   WON'T IMPLEMENT (net-negative).** The proposal (from the generic "deepcopy-under-lock is
   an anti-pattern" guidance) does not hold here: `src/pits_mras/inference/parallel.py:153–157`
   deepcopies the critic under `_critic_lock` as a *deliberate, documented* choice, and the
   heavy IRL gradient step already runs off-lock (the double-buffer point). The critic is a
   tiny quadratic head, so the under-lock copy is microseconds; moving it off-lock would churn
   reviewed concurrency code for no measurable gain. The second half of the generic advice —
   "readers use `torch.inference_mode()`" — is actively wrong here: the control thread's
   costate head needs `autograd.grad` for ∇V̂ (`realtime.py` wraps the step in
   `enable_grad()`), so `inference_mode` would break it. Left as-is, consistent with the
   repo's pattern of recording evaluated-but-declined changes.

### 2.3 Simplicities

8. **Adaptive / causal loss weighting.**
   `src/pits_mras/config.py:75–93` hardcodes fixed `lambda_*`; `training/cotrain.py`
   additionally bakes in a literal `0.1` CBF weight; there is no adaptive weighting. Adopt
   causal time-weighting (near-free, purpose-built for time-dependent systems — the IRL loss
   is already a time-windowed trapezoidal integral) and/or ReLoBRaLo (best accuracy/overhead,
   no extra backward passes). Eliminates the manual weight-tuning that is the leading PINN
   failure cause, and the magic `0.1`.
   *Sources:* causal training (arXiv:2203.07404); ReLoBRaLo (arXiv:2110.09813); RBA
   (arXiv:2307.00379). *Effort:* **S–M.** (Land #9 first.)

9. **Decompose `cotraining_loop` and unify on the existing `TotalLoss` registry.**
   `src/pits_mras/training/cotrain.py:87–360` is one ~270-line function that builds `L_total`
   inline, while `src/pits_mras/losses/__init__.py:61–87` already defines a `TotalLoss`
   registry that `cotrain` does not use. Extract composable phase functions (PITNN step /
   critic step / plant advance), route aggregation through `TotalLoss` (removing the duplicate
   path), and delete the dead `n_heads` parameter declared but never referenced in
   `src/pits_mras/models/attention.py:43–127`. Makes #8 a one-line drop-in.
   *Source:* loss-registry composition for research libraries. *Effort:* **M.**

10. **Tooling / typing pass: Ruff + jaxtyping + fail-loud config validation.**
    (a) Replace flake8 + black + isort with **Ruff** (check + format), keeping mypy;
    (b) add **jaxtyping** shape annotations on control-math signatures
    (`Float[Tensor, "batch state"]`) where silent shape bugs hide;
    (c) make `config.py` `from_yaml` (`src/pits_mras/config.py:158–170`) **error on unknown
    keys** instead of silently ignoring them. *Source:* Ruff complements (does not replace)
    mypy; torchtyping is deprecated — use jaxtyping. *Effort:* **S.**

**Suggested quick-win order** (impact ÷ effort): #4 → #10 → #3 → #5 → #8 → #7, with
#1 → #6 → #2 as the larger capability arc, and #9 landed before #8.

---

## 3. Known Gaps / Deferred (pre-existing)

Genuine un-done items carried over from the original build plan, re-verified against the
source on 2026-06-05:

- **G8 — MIMO control input is simplified in the decoder.** `f_ctrl = B(x_p) · u.sum(...)`
  (`src/pits_mras/models/decoders.py:204–207`, annotated "MIMO-simplified per IP §5.2 / G8").
  Generalize to a proper `B u` product for true multi-input plants.
- **Connection 5 (SAC / max-entropy RL)** — no dedicated module. (The `entropy` term in
  `models/attention.py` is attention regularization, unrelated to SAC.)
- **Connection 9 (TD-MPC2 / learned-model planning)** — no dedicated module.
- **G7 — no `data/` module.** `training/pretrain.py` and `training/irl_trainer.py` assume
  "trajectory data" but there is no dataset format / generator / loader; data is synthesized
  inline in the co-training loop. A `data/` package would be needed for real-trajectory
  training.

---

## 4. Excluded — Already Implemented (honesty note)

The SOTA survey surfaced these, but the codebase already has them — proposing them would be
hallucinated novelty:

- **Integral, model-free IRL Bellman loss** — already implemented (`src/pits_mras/losses/irl.py`,
  trapezoidal integral, drift `A` absent).
- **Persistence-of-excitation monitoring** — already implemented (`src/pits_mras/utils/pe_monitor.py`).
- **Eigenvalues-only PD check (avoids eigenvector-backprop NaNs)** — already used
  (`src/pits_mras/models/critic.py` `positivity_loss`, `torch.linalg.eigvalsh`).
- **float64 analytic core** — already satisfied (SciPy Riccati/Lyapunov/Schur run float64; init-only).
- **Structure-preserving port-Hamiltonian decoder** — already implemented; a GENERIC/GFINN
  thermodynamic extension is a possible *future* capability but overlaps the existing decoder
  and was held back to keep §2 to ten actionable items.

---

*Companion to [`architecture/ARCHITECTURE.md`](architecture/ARCHITECTURE.md) (authoritative
layout/identity map). [`../CHANGELOG.md`](../CHANGELOG.md) is the history of record for shipped
work. If a file path here ever diverges from `architecture/ARCHITECTURE.md`, that file governs
and this one must be corrected.*
