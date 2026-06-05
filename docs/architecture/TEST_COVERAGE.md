# Test Coverage Analysis

**Generated**: 2026-06-05

## Summary

| Metric | Count |
|--------|-------|
| Total Source Files | 44 |
| Total Test Files | 29 |
| Source Files with Tests | 36 |
| Source Files without Tests | 8 |
| Coverage | 81.8% |

---

## Source Files Without Test Coverage

8 source files are not imported by any test:

- `examples/autonomous_vehicle.py`
- `examples/building_hvac.py`
- `examples/pcml_heat_diffusion.py`
- `examples/plants.py`
- `examples/robotic_manipulator.py`
- `setup.py`
- `src/pits_mras/controllers/__init__.py`
- `src/pits_mras/inference/__init__.py`

---

## Source Files With Test Coverage

| Source File | Test Files |
|-------------|------------|
| `src/pits_mras/__init__.py` | `test_imports.py` |
| `src/pits_mras/config.py` | `test_config.py`, `test_inference.py`, `test_losses.py`, `test_models.py`, `test_pcml_integration.py`, `test_smoke.py`, `test_training.py` |
| `src/pits_mras/constraints/__init__.py` | `test_imports.py`, `test_pcml_constraints.py`, `test_pcml_hard.py`, `test_pcml_integration.py`, `test_pcml_jacobian_vectorized.py`, `test_pcml_soft.py` |
| `src/pits_mras/constraints/base.py` | `test_imports.py`, `test_pcml_constraints.py`, `test_pcml_hard.py`, `test_pcml_integration.py`, `test_pcml_jacobian_vectorized.py`, `test_pcml_soft.py` |
| `src/pits_mras/constraints/mechanical.py` | `test_imports.py`, `test_pcml_constraints.py`, `test_pcml_hard.py`, `test_pcml_integration.py`, `test_pcml_jacobian_vectorized.py`, `test_pcml_soft.py` |
| `src/pits_mras/constraints/thermal.py` | `test_imports.py`, `test_pcml_constraints.py`, `test_pcml_hard.py`, `test_pcml_integration.py`, `test_pcml_jacobian_vectorized.py`, `test_pcml_soft.py` |
| `src/pits_mras/controllers/mras.py` | `test_controllers.py`, `test_identity_costate.py`, `test_imports.py`, `test_inference.py`, `test_pcml_integration.py`, `test_smoke.py`, `test_training.py` |
| `src/pits_mras/controllers/reference_models.py` | `test_controllers.py`, `test_identity_costate.py`, `test_identity_lyapunov_value.py`, `test_imports.py`, `test_inference.py`, `test_pcml_integration.py`, `test_safety.py`, `test_smoke.py`, `test_training.py` |
| `src/pits_mras/controllers/safety.py` | `test_imports.py`, `test_safety.py` |
| `src/pits_mras/inference/parallel.py` | `test_inference.py` |
| `src/pits_mras/inference/realtime.py` | `test_imports.py`, `test_inference.py`, `test_pcml_integration.py`, `test_smoke.py` |
| `src/pits_mras/losses/__init__.py` | `test_losses.py`, `test_pcml_integration.py` |
| `src/pits_mras/losses/adaptive_weighting.py` | `test_adaptive_weighting.py` |
| `src/pits_mras/losses/hjb.py` | `test_losses.py`, `test_pcml_integration.py` |
| `src/pits_mras/losses/irl.py` | `test_irl.py`, `test_losses.py`, `test_pcml_integration.py` |
| `src/pits_mras/losses/physics.py` | `test_losses.py`, `test_pcml_integration.py` |
| `src/pits_mras/losses/stability.py` | `test_losses.py`, `test_pcml_integration.py` |
| `src/pits_mras/losses/temporal.py` | `test_losses.py`, `test_pcml_integration.py` |
| `src/pits_mras/models/__init__.py` | `test_inference.py`, `test_smoke.py`, `test_training.py` |
| `src/pits_mras/models/attention.py` | `test_inference.py`, `test_models.py`, `test_smoke.py`, `test_training.py` |
| `src/pits_mras/models/critic.py` | `test_hinf.py`, `test_identity_costate.py`, `test_identity_lyapunov_value.py`, `test_imports.py`, `test_inference.py`, `test_irl.py`, `test_losses.py`, `test_models.py`, `test_smoke.py`, `test_training.py` |
| `src/pits_mras/models/decoders.py` | `test_inference.py`, `test_models.py`, `test_smoke.py`, `test_training.py` |
| `src/pits_mras/models/koopman.py` | `test_inference.py`, `test_koopman.py`, `test_smoke.py`, `test_training.py` |
| `src/pits_mras/models/lagrangian_head.py` | `test_imports.py`, `test_pcml_integration.py`, `test_pcml_soft.py` |
| `src/pits_mras/models/pcml.py` | `test_imports.py`, `test_pcml_hard.py`, `test_pcml_integration.py`, `test_pcml_jacobian_vectorized.py`, `test_pcml_soft.py` |
| `src/pits_mras/models/pitnn.py` | `test_imports.py`, `test_inference.py`, `test_models.py`, `test_pcml_integration.py`, `test_smoke.py`, `test_training.py` |
| `src/pits_mras/training/__init__.py` | `test_imports.py`, `test_smoke.py` |
| `src/pits_mras/training/cotrain.py` | `test_imports.py`, `test_pcml_integration.py`, `test_smoke.py`, `test_training.py` |
| `src/pits_mras/training/irl_trainer.py` | `test_identity_lyapunov_value.py`, `test_imports.py`, `test_smoke.py`, `test_training.py` |
| `src/pits_mras/training/pretrain.py` | `test_imports.py`, `test_smoke.py`, `test_training.py` |
| `src/pits_mras/utils/__init__.py` | `test_lyapunov_utils.py` |
| `src/pits_mras/utils/diagnostics.py` | `test_diagnostics.py` |
| `src/pits_mras/utils/hamiltonian.py` | `test_hamiltonian_utils.py` |
| `src/pits_mras/utils/lyapunov.py` | `test_controllers.py`, `test_differentiable_riccati.py`, `test_hinf.py`, `test_identity_costate.py`, `test_identity_lyapunov_value.py`, `test_lyapunov_utils.py` |
| `src/pits_mras/utils/pe_monitor.py` | `test_pe_monitor.py` |
| `src/pits_mras/utils/uq.py` | `test_uq.py` |
