"""Tests for the Phase-7 runnable examples (IP §10).

Each example exposes a headless ``run(steps=..., show=False) -> dict`` that
builds the full PITNN -> MRAS -> RealtimeInferenceEngine stack, runs the closed
loop, and returns finite metrics plus a matplotlib ``Figure``. These tests run
the loops headless (Agg backend, tiny step counts) and assert no exception,
finite metrics, and a returned figure.

Owning phase: Phase 7 (Examples).
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
from typing import Any

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402  (after backend selection)

_EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent.parent / "examples"
_EXAMPLE_MODULES = [
    "robotic_manipulator",
    "autonomous_vehicle",
    "building_hvac",
]


def _import_run(module_name: str) -> Any:
    """Load an example module from file and return its ``run`` callable."""
    script_path = _EXAMPLES_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(
        f"_pits_example_run_{module_name}", script_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "run"), f"examples.{module_name} missing run()"
    assert hasattr(mod, "main"), f"examples.{module_name} missing main()"
    return mod.run


def _assert_finite_metrics(metrics: dict) -> None:
    """Every numeric / list-of-numeric metric value must be finite."""
    assert isinstance(metrics, dict)
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            assert math.isfinite(float(value)), f"non-finite metric {key!r}={value!r}"
        elif isinstance(value, list):
            for elem in value:
                if isinstance(elem, (int, float)) and not isinstance(elem, bool):
                    assert math.isfinite(float(elem)), f"non-finite in {key!r}"


@pytest.mark.parametrize("module_name", _EXAMPLE_MODULES)
def test_example_run_headless(module_name: str) -> None:
    """``run`` executes headless, returns finite metrics and a Figure."""
    run = _import_run(module_name)
    out = run(steps=12, show=False)
    assert isinstance(out, dict)
    assert "figure" in out, f"{module_name}.run did not return a 'figure'"
    assert isinstance(out["figure"], Figure)
    _assert_finite_metrics(out)


@pytest.mark.parametrize("module_name", _EXAMPLE_MODULES)
def test_example_run_returns_steps_series(module_name: str) -> None:
    """The returned metrics expose an ``error_norm`` series of requested length."""
    run = _import_run(module_name)
    out = run(steps=10, show=False)
    assert "error_norm" in out, f"{module_name}.run missing 'error_norm' series"
    assert isinstance(out["error_norm"], list)
    assert len(out["error_norm"]) == 10


def test_pcml_heat_diffusion_hard_projection() -> None:
    """The coordinate-bearing PCML demo: soft training reduces the heat-equation
    violation, and the hard KKT projection drives it to ~0 (real x, t, autodiff
    derivatives)."""
    run = _import_run("pcml_heat_diffusion")
    out = run(steps=30, show=False)
    assert "figure" in out and isinstance(out["figure"], Figure)
    _assert_finite_metrics(out)
    # Soft training fits the data (data loss falls)...
    assert out["data_loss"][-1] < out["data_loss"][0]
    # ...and the hard KKT projection drives the heat-equation violation to ~0.
    assert out["violation_after_projection"] < out["violation_before_projection"]
    assert out["violation_after_projection"] < 1e-4
