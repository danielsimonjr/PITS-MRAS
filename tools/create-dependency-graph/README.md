# create-dependency-graph

A standalone **Python** utility that scans a Python codebase and generates
dependency + test-coverage documentation under `docs/architecture/`. (Ported
from the original TypeScript tool to support Python projects natively, with no
Node/`tsx` toolchain required.)

## Usage

Run it from the project root with the repo's own Python:

```bash
# Analyze the current project (src/ + examples/ + setup.py), incl. test coverage
python tools/create-dependency-graph/create_dependency_graph.py --include-tests
```

Options:

- `--root=<path>` (or a positional path) — project root to analyze (default: current directory)
- `--exclude=<a,b,c>` — replace the default skip list of directory names
- `--also-exclude=<a,b>` — add directory names to the default skip list
- `--include-tests`, `-t` — include test files (`test_*.py`, `*_test.py`, `tests/`, `conftest.py`) in the analysis; drives `TEST_COVERAGE.md` and makes unused-export detection test-aware
- `--help`, `-h` — show usage

It reads `<root>/setup.py` and/or `<root>/pyproject.toml` for the project
name/version (and any console-script entry points), discovers the importable
**package roots** from the filesystem (e.g. `src/pits_mras` → package
`pits_mras`), then scans the whole root for first-party `.py` source, skipping a
default set of non-source directories:

```
.git, .github, .claude, node_modules, dist, build, .eggs, __pycache__,
.pytest_cache, .mypy_cache, .ruff_cache, htmlcov, .tox, .venv, venv, env,
docs, tools
```

`examples/` is **included** as first-party source; `docs/` (this tool's own
output) and `tools/` (the tool itself) are excluded.

## Output

Written to `<root>/docs/architecture/`:

- `DEPENDENCY_GRAPH.md` — human-readable graph (modules, imports/exports, Mermaid diagram, stats)
- `dependency-graph.json` — full machine-readable graph
- `dependency-graph.yaml` — compact YAML form (requires PyYAML; skipped if unavailable)
- `dependency-summary.compact.json` — minified summary for LLM consumption
- `TEST_COVERAGE.md` + `test-coverage.json` — file-level test coverage (with `--include-tests`)
- `unused-analysis.md` — potentially unused files and exports

## Python concept mapping

The original tool targeted TypeScript; this port maps the concepts to Python:

| Concept | TypeScript | Python (this tool) |
|---|---|---|
| Imports | `import { x } from './m'` | `import m` / `from .m import x` (incl. parenthesized multi-line) |
| Internal deps | relative `./` imports | relative (`.`/`..`) **and** absolute intra-package imports, resolved to file paths |
| Exports | `export` keyword | `__all__` + public top-level `def`/`class`/`CONSTANT` |
| Re-exports | `export * from` | `from .mod import x` inside `__init__.py` (barrels), relative or absolute |
| Type-only imports | `import type` | imports guarded by `if TYPE_CHECKING:` (no runtime cycle) |
| Interfaces / enums | `interface` / `enum` | `class X(Protocol\|ABC)` / `class X(Enum)` |
| Entry points | shebang, `index.ts`, npm scripts | `__main__` blocks, `__init__.py`, `setup.py`, console-scripts, `scripts/`, `examples/` |

## Features

- Parses Python imports (relative + absolute, parenthesized multi-line) and
  classifies them as internal / standard-library / third-party
- Resolves internal imports to root-relative file paths (module *or* package)
- Extracts exports (`__all__`, public defs/classes/constants); detects
  `Protocol`/`ABC` and `Enum` subclasses
- Recognizes barrel re-exports in `__init__.py`
- Detects circular dependencies, separating runtime from `TYPE_CHECKING`-only cycles
- Flags potentially unused files and exports (test imports count as usage with `--include-tests`)
- Generates statistics (file/module count, LOC, exports, classes, etc.)
- Produces human-readable Markdown plus machine-readable JSON/YAML and an LLM-compact summary

## Setup

Pure standard library, except the optional `dependency-graph.yaml` output, which
uses **PyYAML** (already a project runtime dependency here). No extra install is
needed. Tests live alongside the tool:

```bash
pytest tools/create-dependency-graph/        # run the tool's unit tests
```
