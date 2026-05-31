"""Smoke test (IP §11.6): end-to-end, no crash / no NaN.

Targets the full stack (``pitnn`` -> controller -> inference) plus the example
scripts. Owning phase: Phase 5 / Phase 6 per ROADMAP.md (authored alongside its
target phase; §11 catalogs it under "Phase 8").

This file provides:
  * ``test_package_imports`` / ``test_example_script_imports`` -- ACTIVE Phase-0
    scaffold gates proving the package and the three ``examples/`` scripts import
    without error (complementing the exhaustive per-module check in
    ``test_imports.py``).
  * the three verbatim §11.6 placeholders (skipped until implemented):
    ``test_full_forward_pass_no_crash``, ``test_pretrain_one_epoch``,
    ``test_cotrain_one_episode``.
"""

import importlib
import importlib.util
import pathlib

import pytest

_EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent.parent / "examples"
_EXAMPLE_SCRIPTS = ["robotic_manipulator", "autonomous_vehicle", "building_hvac"]


def test_package_imports() -> None:
    """The top-level package imports and exposes its version."""
    pkg = importlib.import_module("pits_mras")
    assert pkg.__version__ == "1.0.0"


@pytest.mark.parametrize("script_name", _EXAMPLE_SCRIPTS)
def test_example_script_imports(script_name: str) -> None:
    """Each example script imports without error."""
    script_path = _EXAMPLES_DIR / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(
        f"_pits_example_{script_name}", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


@pytest.mark.skip(reason="phase 6 not implemented")
def test_full_forward_pass_no_crash() -> None:
    """Run 10 steps of RealtimeInferenceEngine with no exceptions / NaN."""


@pytest.mark.skip(reason="phase 5 not implemented")
def test_pretrain_one_epoch() -> None:
    """One pretrain epoch produces finite (non-NaN) losses."""


@pytest.mark.skip(reason="phase 5 not implemented")
def test_cotrain_one_episode() -> None:
    """One co-training episode produces finite (non-NaN) losses."""
