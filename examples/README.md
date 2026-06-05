# PITS-MRAS Examples

Runnable closed-loop demos of the PITS-MRAS stack (PITNN → MRAS controller →
CLF-CBF safety filter → `RealtimeInferenceEngine`). Each script exposes
`run(steps=..., show=False) -> dict` (headless, returns metrics + a matplotlib
`Figure`) and a `main()` entry point, and is import-safe (all sim/plot work is
inside `run()`/`main()`).

The plants are **nonlinear** (see [`plants.py`](./plants.py)); each linearizes to
the controller's reference model near the operating point, so the LQR/CBF
controller stays stabilizing while the nonlinearity exercises model-mismatch
robustness. *Note: the demos are illustrative — not validated against hardware.*

## Examples

### `robotic_manipulator.py`
2-DOF manipulator joint tracking a sinusoidal reference. Plant: nonlinear
`pendulum_step` (sin-gravity). Four-panel figure: tracking error, `V̂`,
CBF-activation, and the offline-IRL critic convergence curve (panel d).

### `autonomous_vehicle.py`
Lane-hold under a strong lateral wind gust, with vs. without the CBF safety
filter. Plant: `lateral_tyre_step` (single-track lateral dynamics with `tanh`
tyre-force saturation).

### `building_hvac.py`
Zone-temperature setpoint tracking vs. a proportional baseline. Plant:
`rc_thermal_step` (2-node RC building-thermal network + saturated heater); the
example's reference model is the RC linearization.

### `pcml_heat_diffusion.py`
Coordinate-bearing hard-PCML demo: a small `T(x, t)` MLP on the 1-D heat equation
with genuine `(x, t, ∂)` autodiff derivatives; soft PCML reduces the residual and
the KKT projection drives the point-wise violation to ~0.

## Running

```bash
pip install -e ..          # install the package (from the repo root)
python robotic_manipulator.py   # runs main() -> displays/saves the figure
```

Or headless from Python:

```python
from importlib import import_module
out = import_module("robotic_manipulator").run(steps=100, show=False)
print(out.keys())          # error_norm, v_hat, cbf_active, figure, ...
```

The example smoke tests live in [`../tests/test_examples.py`](../tests/test_examples.py)
(headless `run()` per example) and the plant physics in
[`../tests/test_example_plants.py`](../tests/test_example_plants.py).

## Contributing Examples

Have an interesting use case? Contributions welcome — see
[CONTRIBUTING.md](../CONTRIBUTING.md).
