# PITS-MRAS Source Code

The implemented `pits_mras` package. See
[`docs/architecture/`](../docs/architecture/) for the graph-backed component /
API / data-flow docs, and [`CHANGELOG.md`](../CHANGELOG.md) for release history.

## Structure

```
pits_mras/
├── __init__.py                     # public API + __version__
├── config.py                       # PITSMRASConfig / NetworkConfig / PhysicsConfig / LossConfig / ...
├── models/
│   ├── pitnn.py                    # Physics-Informed Temporal NN (LSTM + attention + decoder)
│   ├── attention.py                # physics-informed multi-head attention
│   ├── decoders.py                 # port-Hamiltonian decoder (HamiltonianNet, DissipationNet); MIMO control via B @ u
│   ├── critic.py                   # QuadraticCritic, CostateHead, AdversaryHead (H∞)
│   ├── adversary.py                # NeuralAdversary (learned H∞ disturbance)
│   ├── pcml.py                     # PCML: SoftPCMLLoss, KKTProjectionLayer, PCMLModule
│   ├── lagrangian_head.py          # Lagrangian-multiplier head
│   ├── koopman.py                  # KoopmanLiftingModel + koopman_loss (deep Koopman lifting)
│   ├── generic.py                  # GFINNDecoder (GENERIC structure-preserving dynamics)
│   ├── sac.py                      # GaussianPolicy + TwinQCritic (SAC)
│   └── tdmpc.py                    # WorldModel + MPPIPlanner (TD-MPC2)
├── controllers/
│   ├── mras.py                     # MRASController
│   ├── reference_models.py         # LinearReferenceModel
│   ├── safety.py                   # CLF-CBF safety filter
│   └── koopman_control.py          # KoopmanLQRController
├── constraints/                    # PCML physics constraints
│   ├── base.py                     # PhysicsConstraints ABC + ConstraintSpec
│   ├── mechanical.py               # MechanicalDAE
│   └── thermal.py                  # HeatConductionDAE
├── losses/
│   ├── physics.py temporal.py stability.py   # the loss families
│   ├── irl.py                      # Integral-RL Bellman loss
│   ├── hjb.py                      # HJB residual loss
│   ├── adaptive_weighting.py       # automatic loss-term balancing
│   └── __init__.py                 # TotalLoss aggregator
├── training/
│   ├── pretrain.py                 # Algorithm 2: 3-stage curriculum pre-training
│   ├── cotrain.py                  # Algorithm 3: closed-loop actor-critic co-training
│   ├── irl_trainer.py             # offline IRL critic fitting
│   ├── hinf_minmax.py             # H∞ neural adversarial min-max training loop
│   ├── sac.py                      # SAC training loop
│   └── tdmpc.py                    # TD-MPC2 training loop
├── inference/
│   ├── realtime.py                 # RealtimeInferenceEngine (thread-safe single loop)
│   └── parallel.py                 # ParallelInferenceEngine (3-thread deployment scaffold)
├── data/
│   └── trajectory.py               # synthetic trajectory generation + TrajectoryDataset (opt-in)
└── utils/
    ├── lyapunov.py                 # Lyapunov / Riccati engine (solve_lyapunov/care/gare, kleinman, differentiable CARE/GARE)
    ├── hamiltonian.py              # port-Hamiltonian utilities
    ├── pe_monitor.py               # persistency-of-excitation monitor
    ├── uq.py                       # uncertainty quantification (deep ensembles + conformal intervals)
    ├── diagnostics.py              # rollout diagnostics (energy drift, valid prediction time, spectral radius)
    └── linearization.py            # local linearization helpers
```

## Status

- ✅ **Implemented** — the full build (config/math → models → losses → controllers
  → training → inference → examples → tests → CI), the PCML physics-constraint
  layer, H∞ robust control (analytic GARE core + neural min-max loop), deep
  Koopman lifting + Koopman-LQR control, GENERIC/GFINN, SAC and TD-MPC2 learners,
  uncertainty quantification, and rollout diagnostics. Gates: `pytest` green,
  `ruff check` + `ruff format` + `mypy` clean, dependency graph 0 circular /
  0 unused.
- The rigorously-verified part is the *mathematical core* (the identities); the
  bundled example plants are illustrative nonlinear surrogates, not
  hardware-validated.

## Contributing

See the main [README.md](../README.md) and [CONTRIBUTING.md](../CONTRIBUTING.md).
