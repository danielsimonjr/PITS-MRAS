# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`docs/ARCHITECTURE.md`** — design/architecture blueprint distilled from the two
  design PDFs in `docs/` (the *Mathematical and Architectural Blueprint* and the
  *Complete Implementation Plan*). Documents the three-paradigm merger (PINN +
  time-series deep learning + MRAS), the canonical module layout, the mapping of the
  ten RL/optimal-control identities to owning modules, the new loss terms / network
  heads / re-derived unified adaptation law, the data flow (with a mermaid diagram),
  the training & inference pipeline, and the stability/safety/testing strategy.
- **`docs/ROADMAP.md`** — phased roadmap operationalizing the Implementation Plan:
  9 build phases (Foundation → CI/CD) grouped into 4 milestones, each with
  deliverables, dependencies, acceptance gates, and checkbox task lists. Priorities
  follow the blueprint's highest-leverage-first ordering (Integral-RL policy
  evaluation + CLF-CBF-QP safety filter). Source gaps/conflicts are flagged (G0–G9),
  including the Blueprint-vs-Plan disagreement on the third network head
  (adversary/H∞ vs costate) and the `setup.py`/`requirements.txt` naming conflict.
- **`src/pits_mras/` package scaffold** — package tree matching the architecture's
  canonical layout: `models/`, `controllers/`, `losses/`, `training/`, `inference/`,
  `utils/`, plus `config.py`. Each module is a documented stub (purpose + owning
  phase + relevant identity + `TODO(phase-N)`); modules with a named API expose a
  stub class raising `NotImplementedError`. `config.py` uses stdlib `dataclasses`
  (the design's stated choice; no pydantic dependency introduced).
- **Test skeleton** — `tests/test_imports.py` smoke test importing every module in
  the package, plus the six identity/safety/model/IRL/smoke test files from the plan
  with their mandated test names as `@pytest.mark.skip` placeholders pending
  implementation. Also three `examples/` stubs.
- **CI + tooling** — GitHub Actions workflow (`.github/workflows/ci.yml`) running
  flake8, mypy, and pytest+coverage across Python 3.10–3.12; `setup.cfg` (flake8
  config); `pyproject.toml` (black/isort/mypy/pytest config); `requirements-dev.txt`
  (dev toolchain).

### Notes

- This is a **foundation scaffold**: modules raise `NotImplementedError` or are
  documented placeholders. Implementation proceeds per `docs/ROADMAP.md`.
- Verified gates on this scaffold: `flake8 src tests` → 0; `mypy src` → 0;
  `pytest` → 33 passed, 16 skipped; `import pits_mras` → version 1.0.0.
- The CI install step is intentionally `pip install -e . --no-deps`: the scaffold
  stubs import only the standard library, so the import smoke test passes without
  the heavy ML stack. Switch to `pip install -e .[dev]` once modules begin importing
  torch/numpy (Phase 2+).
- **Deferred decision (ROADMAP gap G2):** `setup.py` is left unchanged
  (`name="pits-mras"`, `version="1.0.0"`); the Implementation Plan instead specifies
  `name="pits_mras"`, `version="0.1.0"` and a dependency overhaul. This packaging
  reconciliation is an ADR-level decision left for the project owner.
- **Python floor:** the Implementation Plan stated a 3.9 baseline, but the current
  mypy release dropped `python_version = 3.9` support and `torch>=2.0.0` requires
  3.10+, so the CI matrix and tool configs use **3.10/3.11/3.12**.
