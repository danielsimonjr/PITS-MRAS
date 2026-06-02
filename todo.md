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

## In progress — PCML feature (TDD, faithful to DAE-HardNet)

- [ ] **Constraints library** `src/pits_mras/constraints/{base,mechanical,thermal}.py`
  — `PhysicsConstraints` ABC + `ConstraintSpec`, `MechanicalDAE` (Euler-Lagrange,
  optional holonomic), `HeatConductionDAE` (1-D heat eq).
- [ ] **PCML core** `src/pits_mras/models/pcml.py` — `SoftPCMLLoss`,
  `TaylorNeighborhoodApproximation`, `KKTProjectionLayer` (Fischer-Burmeister +
  differentiable Newton), `PCMLModule` (soft/hard dynamic activation at `η`).
- [ ] **Lagrangian head** `src/pits_mras/models/lagrangian_head.py` —
  `LagrangianMultiplierHead` (Newton warm-start multipliers).
- [ ] **Integration** — `PCMLConfig` in `config.py`; `lam_hat` output + head in
  `pitnn.py`; `pcml` component(s) in `TotalLoss`; dynamic activation + mode switch
  in `cotrain.py`; projection-bypass in `realtime.py`. PCML opt-in (backward compat).
- [ ] **Tests** — `tests/test_pcml_soft.py`, `tests/test_pcml_hard.py`,
  `tests/test_pcml_integration.py`.
- [ ] **Docs** — PCML section in `docs/` architecture + `PITS-MRAS_FINAL_SUMMARY.md`;
  README; version bump (target v0.3.0) at release.

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
