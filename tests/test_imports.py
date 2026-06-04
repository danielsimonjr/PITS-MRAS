"""Smoke test: import every module in the ``pits_mras`` package.

Phase 0 (Scaffold & Tooling). Proves the entire module tree enumerated in
``docs/ARCHITECTURE.md`` §2 imports without error. This is the GREEN gate for
the scaffold and must keep passing as later phases (1-7) fill in the stubs.
"""

import importlib

import pytest

# Every importable module path listed in docs/ARCHITECTURE.md §2 (canonical
# tree). Note: there is intentionally NO ``registry.py`` (see ARCHITECTURE.md
# §5 -- it is explicitly not a planned module).
ALL_MODULES = [
    "pits_mras",
    "pits_mras.config",
    "pits_mras.models",
    "pits_mras.models.attention",
    "pits_mras.models.decoders",
    "pits_mras.models.critic",
    "pits_mras.models.pitnn",
    "pits_mras.controllers",
    "pits_mras.controllers.reference_models",
    "pits_mras.controllers.safety",
    "pits_mras.controllers.mras",
    "pits_mras.losses",
    "pits_mras.losses.physics",
    "pits_mras.losses.temporal",
    "pits_mras.losses.stability",
    "pits_mras.losses.irl",
    "pits_mras.losses.hjb",
    "pits_mras.training",
    "pits_mras.training.pretrain",
    "pits_mras.training.cotrain",
    "pits_mras.training.irl_trainer",
    "pits_mras.inference",
    "pits_mras.inference.realtime",
    "pits_mras.inference.parallel",
    "pits_mras.utils",
    "pits_mras.utils.lyapunov",
    "pits_mras.utils.hamiltonian",
    "pits_mras.utils.pe_monitor",
]


@pytest.mark.parametrize("module_name", ALL_MODULES)
def test_module_imports(module_name: str) -> None:
    """Each canonical module imports cleanly."""
    importlib.import_module(module_name)


def test_version() -> None:
    """Top-level package exposes a version string matching setup.py."""
    import pits_mras

    assert pits_mras.__version__ == "0.4.1"
