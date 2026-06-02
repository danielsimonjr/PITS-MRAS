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

## Deferred / future (documented, low priority)

- **Synthetic-loop PCML inputs are placeholders**: `cotrain`/`realtime` pass
  zeros for the constraint inputs `x`/`t` and derivative variables `d`, because
  the synthetic plant has no spatial/temporal coordinates. A real plant with
  genuine `(x, t, ∂)` would make the loop-level PCML loss/projection physically
  meaningful (the standalone `PCMLModule` is already fully exercised on real DAEs).
- **2nd-order Taylor (`order=2`) + `MechanicalDAE` holonomic path** are
  implemented but only lightly tested vs the unconstrained/`HeatConductionDAE`
  cases.
- Pre-existing v0.2.0 TODOs still open: vacuous AV-CBF demo margin, untrained
  manipulator critic in examples, H∞ adversary head (gap G1).

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
