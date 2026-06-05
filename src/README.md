# PITS-MRAS Source Code

The implemented `pits_mras` package (released through **v0.4.5**). See
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
│   ├── decoders.py                 # port-Hamiltonian decoder (HamiltonianNet, DissipationNet)
│   ├── critic.py                   # QuadraticCritic, CostateHead, AdversaryHead (H∞)
│   ├── pcml.py                     # PCML: SoftPCMLLoss, KKTProjectionLayer, PCMLModule
│   └── lagrangian_head.py          # Lagrangian-multiplier head
├── controllers/
│   ├── mras.py                     # MRASController
│   ├── reference_models.py         # LinearReferenceModel
│   └── safety.py                   # CLF-CBF safety filter
├── constraints/                    # PCML physics constraints
│   ├── base.py                     # PhysicsConstraints ABC + ConstraintSpec
│   ├── mechanical.py               # MechanicalDAE
│   └── thermal.py                  # HeatConductionDAE
├── losses/
│   ├── physics.py temporal.py stability.py   # the loss families
│   ├── irl.py                      # Integral-RL Bellman loss
│   ├── hjb.py                      # HJB residual loss
│   └── __init__.py                 # TotalLoss aggregator
├── training/
│   ├── pretrain.py                 # Algorithm 2: 3-stage curriculum pre-training
│   ├── cotrain.py                  # Algorithm 3: closed-loop actor-critic co-training
│   └── irl_trainer.py             # offline IRL critic fitting
├── inference/
│   ├── realtime.py                 # RealtimeInferenceEngine (thread-safe single loop)
│   └── parallel.py                 # ParallelInferenceEngine (3-thread deployment scaffold)
└── utils/
    ├── lyapunov.py                 # Lyapunov / Riccati engine (solve_lyapunov/care/gare, kleinman)
    ├── hamiltonian.py              # port-Hamiltonian utilities
    └── pe_monitor.py               # persistency-of-excitation monitor
```

## Status

- ✅ **Implemented** — all 9 ROADMAP phases + the PCML layer + the v0.4.x feature
  line (released through v0.4.5). Gates: `pytest` green, `flake8` + `mypy` clean,
  dependency graph 0 circular / 0 unused.
- The rigorously-verified part is the *mathematical core* (the identities); the
  bundled example plants are illustrative nonlinear surrogates, not
  hardware-validated.

## Contributing

See the main [README.md](../README.md) and [CONTRIBUTING.md](../CONTRIBUTING.md).
