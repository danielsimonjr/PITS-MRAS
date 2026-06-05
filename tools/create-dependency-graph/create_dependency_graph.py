#!/usr/bin/env python3
"""Generic Python dependency-graph generator.

Scans a Python codebase and generates, under ``docs/architecture/``:

- ``DEPENDENCY_GRAPH.md``           -- human-readable graph + Mermaid diagram
- ``dependency-graph.json``         -- machine-readable
- ``dependency-graph.yaml``         -- compact (if PyYAML is available)
- ``dependency-summary.compact.json`` -- minified summary for LLM consumption
- ``TEST_COVERAGE.md`` / ``test-coverage.json`` (with ``--include-tests``)
- ``unused-analysis.md``            -- potentially unused files / exports

This is a standalone Python port of the TypeScript ``create-dependency-graph``
tool. It is generic and discovers the project structure from the filesystem: it
scans the project root for first-party ``.py`` source, skipping a default set of
non-source directories (VCS, build output, caches, ``docs/`` output, and the
``tools/`` folder itself). Override the skip list with ``--exclude``.

Python concept mapping vs the TS original:
- imports -> ``import x`` / ``from .pkg import a`` (incl. parenthesized multi-line)
- internal deps -> relative imports and absolute intra-package imports, resolved
  to root-relative file paths
- exports -> ``__all__`` plus public top-level ``def`` / ``class`` / constants
- re-exports -> ``from .mod import x`` inside ``__init__.py`` (barrels)
- type-only imports -> imports guarded by ``if TYPE_CHECKING:`` (no runtime cycle)
- entry points -> ``__main__`` blocks, ``__init__.py``, ``setup.py``,
  console-scripts, and ``scripts/`` / ``examples/`` files

Usage: ``python tools/create-dependency-graph/create_dependency_graph.py [--include-tests]``
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Set, Tuple

# Directory names skipped at any depth: VCS, build output, caches, virtualenvs,
# generated docs (incl. this tool's own reports), and tooling that is not
# first-party application source.
DEFAULT_EXCLUDE_DIRS = [
    ".git",
    ".github",
    ".claude",
    "node_modules",
    "dist",
    "build",
    ".eggs",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
    ".tox",
    ".venv",
    "venv",
    "env",
    "docs",  # generated output (incl. this tool's own reports)
    "tools",  # this tool and its siblings are not application source
]

# Python standard-library module names (top-level). Uses the interpreter's own
# list when available (3.10+), with a small fallback for older runtimes.
try:
    _STDLIB = set(sys.stdlib_module_names)  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - very old Python
    _STDLIB = {
        "abc", "argparse", "ast", "asyncio", "collections", "contextlib", "copy",
        "dataclasses", "datetime", "enum", "functools", "io", "itertools", "json",
        "logging", "math", "os", "pathlib", "re", "sys", "threading", "time",
        "typing", "warnings", "weakref",
    }


@dataclass
class InternalDep:
    file: Optional[str]  # resolved root-relative path (None if unresolved)
    module: str  # raw dotted module string (for display)
    imports: List[str]
    re_export: bool = False
    type_only: bool = False  # guarded by `if TYPE_CHECKING:`


@dataclass
class ExternalDep:
    package: str
    imports: List[str]


@dataclass
class FileExports:
    named: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    interfaces: List[str] = field(default_factory=list)  # Protocol / ABC subclasses
    enums: List[str] = field(default_factory=list)  # Enum subclasses
    functions: List[str] = field(default_factory=list)
    constants: List[str] = field(default_factory=list)  # UPPER_CASE assignments
    re_exported: List[str] = field(default_factory=list)
    declared_all: bool = False  # whether the file defines __all__


@dataclass
class ParsedFile:
    path: str
    name: str
    external_dependencies: List[ExternalDep] = field(default_factory=list)
    stdlib_dependencies: List[ExternalDep] = field(default_factory=list)
    internal_dependencies: List[InternalDep] = field(default_factory=list)
    exports: FileExports = field(default_factory=FileExports)
    description: Optional[str] = None
    is_executable: bool = False  # has a `__main__` block or shebang
    referenced_file_names: List[str] = field(default_factory=list)
    lines: int = 0


# --------------------------------------------------------------------------- #
# Package discovery + import resolution.
# --------------------------------------------------------------------------- #
def discover_package_roots(root: str, exclude: Set[str]) -> Dict[str, str]:
    """Map each top-level importable package name to its root-relative directory.

    A package is a directory with ``__init__.py`` whose parent has none (so it is
    the top of an import path, e.g. ``src/pits_mras`` -> ``pits_mras``).
    """
    roots: Dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude]
        if "__init__.py" in filenames:
            parent = os.path.dirname(dirpath)
            if not os.path.exists(os.path.join(parent, "__init__.py")):
                pkg = os.path.basename(dirpath)
                rel = os.path.relpath(dirpath, root).replace(os.sep, "/")
                roots.setdefault(pkg, rel)
    return roots


def _module_to_relpath(
    module_path: str, package_roots: Dict[str, str], all_files: Set[str]
) -> Optional[str]:
    """Resolve a dotted module (``pkg.sub.mod`` or a relative-resolved one) to a file."""
    parts = module_path.split(".")
    top = parts[0]
    if top not in package_roots:
        return None
    base = package_roots[top]
    sub = "/".join(parts[1:])
    candidate_mod = f"{base}/{sub}.py" if sub else f"{base}.py"
    candidate_pkg = f"{base}/{sub}/__init__.py" if sub else f"{base}/__init__.py"
    if candidate_mod in all_files:
        return candidate_mod
    if candidate_pkg in all_files:
        return candidate_pkg
    return None


def resolve_relative(file_rel: str, level: int, module: str) -> str:
    """Resolve a relative import to an absolute dotted module path.

    ``file_rel`` is the importing file's root-relative path; ``level`` is the
    number of leading dots; ``module`` is the dotted text after the dots.
    """
    # The file's package is its containing directory, expressed as dotted path
    # relative to the discovered package root is handled by the caller; here we
    # work in directory space and return a "<dir>/<module>" pseudo path the
    # caller maps against the filesystem.
    pkg_dir = os.path.dirname(file_rel)
    up = level - 1
    parts = pkg_dir.split("/") if pkg_dir else []
    if up > 0:
        parts = parts[: len(parts) - up] if up <= len(parts) else []
    base = "/".join(parts)
    if module:
        base = f"{base}/{module.replace('.', '/')}" if base else module.replace(".", "/")
    return base


def _resolve_relative_to_file(
    file_rel: str, level: int, module: str, names: List[str], all_files: Set[str]
) -> List[Tuple[Optional[str], str]]:
    """Resolve a relative import to (resolved_path, raw_module) targets."""
    base = resolve_relative(file_rel, level, module)
    results: List[Tuple[Optional[str], str]] = []
    if module:
        mod_file = f"{base}.py"
        pkg_file = f"{base}/__init__.py"
        if mod_file in all_files:
            results.append((mod_file, "." * level + module))
        elif pkg_file in all_files:
            results.append((pkg_file, "." * level + module))
        else:
            results.append((None, "." * level + module))
    else:
        # `from . import a, b` -- each name is a submodule of the base package.
        for nm in names:
            cand_mod = f"{base}/{nm}.py" if base else f"{nm}.py"
            cand_pkg = f"{base}/{nm}/__init__.py" if base else f"{nm}/__init__.py"
            if cand_mod in all_files:
                results.append((cand_mod, "." * level + nm))
            elif cand_pkg in all_files:
                results.append((cand_pkg, "." * level + nm))
            else:
                results.append((None, "." * level + nm))
    return results


# --------------------------------------------------------------------------- #
# Parsing.
# --------------------------------------------------------------------------- #
def _logical_import_lines(content: str) -> List[Tuple[int, str]]:
    """Yield (start_line_index, joined_logical_line) for import statements.

    Joins parenthesized multi-line ``from x import (a, b, c)`` into one logical
    line so a single regex can parse it.
    """
    raw = content.split("\n")
    out: List[Tuple[int, str]] = []
    i = 0
    n = len(raw)
    while i < n:
        line = raw[i]
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            start = i
            buf = line
            # Join continuations: an unclosed paren or a trailing backslash. The
            # paren check uses the code portion only (text before any ``#``), so a
            # parenthesis inside a trailing comment (e.g. ``# noqa (sibling mod)``)
            # does not look like an open paren and swallow the following lines.
            while True:
                code = buf.split("#")[0]
                if not (("(" in code and ")" not in code) or code.rstrip().endswith("\\")):
                    break
                i += 1
                if i >= n:
                    break
                buf = buf.rstrip("\\") + " " + raw[i]
            out.append((start, buf))
        i += 1
    return out


def _typechecking_line_ranges(content: str) -> List[Tuple[int, int]]:
    """Return [start, end) physical-line ranges guarded by ``if TYPE_CHECKING:``."""
    raw = content.split("\n")
    ranges: List[Tuple[int, int]] = []
    for idx, line in enumerate(raw):
        m = re.match(r"^(\s*)if\s+(?:typing\.)?TYPE_CHECKING\s*:", line)
        if not m:
            continue
        indent = len(m.group(1))
        end = idx + 1
        while end < len(raw):
            nxt = raw[end]
            if nxt.strip() == "":
                end += 1
                continue
            cur_indent = len(nxt) - len(nxt.lstrip())
            if cur_indent <= indent:
                break
            end += 1
        ranges.append((idx + 1, end))
    return ranges


def _in_typechecking(line_idx: int, ranges: List[Tuple[int, int]]) -> bool:
    return any(start <= line_idx < end for start, end in ranges)


def _split_import_names(names_blob: str) -> List[str]:
    """Parse the names in ``a, b as c, d`` -> ['a', 'c', 'd'] (alias kept).

    A trailing ``# ...`` comment (e.g. ``foo  # noqa: E402``) is stripped first so
    the comment text is never mistaken for an imported name.
    """
    names_blob = names_blob.split("#", 1)[0]
    out: List[str] = []
    for item in names_blob.replace("(", "").replace(")", "").split(","):
        item = item.strip()
        if not item or item == "*":
            if item == "*":
                out.append("*")
            continue
        # `name as alias` -> the bound name is the alias.
        parts = re.split(r"\s+as\s+", item)
        bound = parts[-1].strip()
        if bound:
            out.append(bound)
    return out


def extract_description(content: str) -> Optional[str]:
    """First non-empty line of the module docstring, if any."""
    m = re.match(r'^\s*[rRbBuU]*("""|\'\'\')(.*?)(\1)', content, re.DOTALL)
    if m:
        body = m.group(2).strip()
        for line in body.split("\n"):
            line = line.strip()
            if line:
                return line
    # Fall back to a leading `# comment`.
    cm = re.search(r"^#\s*(.+)$", content, re.MULTILINE)
    if cm:
        return cm.group(1).strip()
    return None


def parse_file(
    abs_path: str,
    root: str,
    package_roots: Dict[str, str],
    all_files: Set[str],
) -> ParsedFile:
    with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
        content = fh.read()
    rel = os.path.relpath(abs_path, root).replace(os.sep, "/")
    pf = ParsedFile(path=rel, name=os.path.splitext(os.path.basename(abs_path))[0])
    pf.lines = content.count("\n") + 1
    pf.description = extract_description(content)
    pf.is_executable = content.startswith("#!") or bool(
        re.search(r'^\s*if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:', content, re.MULTILINE)
    )
    is_init = pf.name == "__init__"

    # Files referenced by name as string literals (e.g. a spawned child script).
    for m in re.finditer(r"""['"]([\w./-]+\.py)['"]""", content):
        pf.referenced_file_names.append(os.path.basename(m.group(1)))

    tc_ranges = _typechecking_line_ranges(content)

    for start_idx, logical in _logical_import_lines(content):
        type_only = _in_typechecking(start_idx, tc_ranges)
        _parse_import_statement(
            logical, pf, type_only, is_init, package_roots, all_files
        )

    _parse_exports(content, pf, is_init)
    return pf


def _record_internal(
    pf: ParsedFile,
    resolved: Optional[str],
    module: str,
    names: List[str],
    re_export: bool,
    type_only: bool,
) -> None:
    pf.internal_dependencies.append(
        InternalDep(
            file=resolved,
            module=module,
            imports=names or ["(module)"],
            re_export=re_export,
            type_only=type_only,
        )
    )


def _parse_import_statement(
    logical: str,
    pf: ParsedFile,
    type_only: bool,
    is_init: bool,
    package_roots: Dict[str, str],
    all_files: Set[str],
) -> None:
    logical = logical.strip()
    # from <dots><module> import <names>
    m = re.match(r"^from\s+(\.*)([\w.]*)\s+import\s+(.+)$", logical)
    if m:
        dots, module, names_blob = m.group(1), m.group(2), m.group(3)
        level = len(dots)
        names = _split_import_names(names_blob)
        if level > 0:
            targets = _resolve_relative_to_file(pf.path, level, module, names, all_files)
            for resolved, raw_mod in targets:
                _record_internal(pf, resolved, raw_mod, names, is_init, type_only)
            return
        # Absolute import.
        top = module.split(".")[0]
        if top in package_roots:
            resolved = _module_to_relpath(module, package_roots, all_files)
            # An absolute intra-package from-import inside __init__.py is a
            # barrel re-export, just like the relative form.
            _record_internal(pf, resolved, module, names, is_init, type_only)
        elif top in _STDLIB:
            pf.stdlib_dependencies.append(ExternalDep(package=module, imports=names))
        else:
            pf.external_dependencies.append(ExternalDep(package=module, imports=names))
        return
    # import a, b.c as d
    m = re.match(r"^import\s+(.+)$", logical)
    if m:
        for part in m.group(1).split(","):
            part = part.strip()
            if not part:
                continue
            mod = re.split(r"\s+as\s+", part)[0].strip()
            top = mod.split(".")[0]
            if top in package_roots:
                resolved = _module_to_relpath(mod, package_roots, all_files)
                _record_internal(pf, resolved, mod, ["(module)"], False, type_only)
            elif top in _STDLIB:
                pf.stdlib_dependencies.append(ExternalDep(package=mod, imports=["(module)"]))
            else:
                pf.external_dependencies.append(ExternalDep(package=mod, imports=["(module)"]))


def _parse_exports(content: str, pf: ParsedFile, is_init: bool) -> None:
    exp = pf.exports
    # __all__ = [ ... ]  (possibly multi-line)
    am = re.search(
        r"^__all__\s*(?::[^=]+)?=\s*[\[\(](.*?)[\]\)]", content, re.DOTALL | re.MULTILINE
    )
    if am:
        exp.declared_all = True
        for s in re.findall(r"""['"]([\w]+)['"]""", am.group(1)):
            if s not in exp.named:
                exp.named.append(s)

    for m in re.finditer(r"^(?:async\s+)?def\s+(\w+)\s*\(", content, re.MULTILINE):
        name = m.group(1)
        if name.startswith("_"):
            continue
        exp.functions.append(name)
        if name not in exp.named:
            exp.named.append(name)

    for m in re.finditer(r"^class\s+(\w+)\s*(\(([^)]*)\))?\s*:", content, re.MULTILINE):
        name, bases = m.group(1), (m.group(3) or "")
        if name.startswith("_"):
            continue
        if re.search(r"\bEnum\b|\bIntEnum\b|\bStrEnum\b|\bFlag\b", bases):
            exp.enums.append(name)
        elif re.search(r"\bProtocol\b|\bABC\b", bases):
            exp.interfaces.append(name)
        else:
            exp.classes.append(name)
        if name not in exp.named:
            exp.named.append(name)

    # Top-level constants: UPPER_CASE assignments at column 0.
    for m in re.finditer(r"^([A-Z][A-Z0-9_]+)\s*(?::[^=]+)?=", content, re.MULTILINE):
        name = m.group(1)
        if name == "__all__":
            continue
        exp.constants.append(name)
        if name not in exp.named:
            exp.named.append(name)

    # Re-exports: in a barrel __init__.py, every internal from-import is a re-export.
    if is_init:
        for dep in pf.internal_dependencies:
            if dep.re_export:
                for imp in dep.imports:
                    if imp not in ("(module)", "*") and imp not in exp.re_exported:
                        exp.re_exported.append(imp)
                    if imp not in ("(module)", "*") and imp not in exp.named:
                        exp.named.append(imp)

    exp.named = list(dict.fromkeys(exp.named))
    exp.re_exported = list(dict.fromkeys(exp.re_exported))


# --------------------------------------------------------------------------- #
# Analysis.
# --------------------------------------------------------------------------- #
def categorize_files(files: List[ParsedFile]) -> Dict[str, Dict[str, ParsedFile]]:
    modules: Dict[str, Dict[str, ParsedFile]] = {}
    for f in files:
        slash = f.path.rfind("/")
        module = "root" if slash == -1 else f.path[:slash]
        modules.setdefault(module, {})[f.path] = f
    return modules


def detect_circular(files: List[ParsedFile]) -> Dict[str, List[List[str]]]:
    file_paths = {f.path for f in files}
    runtime_graph: Dict[str, List[str]] = {}
    all_graph: Dict[str, List[str]] = {}
    for f in files:
        runtime: List[str] = []
        alld: List[str] = []
        for d in f.internal_dependencies:
            if d.file and d.file in file_paths:
                alld.append(d.file)
                if not d.type_only:
                    runtime.append(d.file)
        runtime_graph[f.path] = runtime
        all_graph[f.path] = alld

    def find_cycles(graph: Dict[str, List[str]]) -> List[List[str]]:
        cycles: List[List[str]] = []
        seen_keys: Set[str] = set()
        visited: Set[str] = set()
        in_stack: Set[str] = set()

        def dfs(node: str, path: List[str]) -> None:
            if node in in_stack:
                start = path.index(node) if node in path else -1
                if start != -1:
                    cycle = path[start:] + [node]
                    key = "->".join(sorted(cycle))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        cycles.append(cycle)
                return
            if node in visited:
                return
            visited.add(node)
            in_stack.add(node)
            path.append(node)
            for nb in graph.get(node, []):
                dfs(nb, path)
            path.pop()
            in_stack.discard(node)

        for node in graph:
            if node not in visited:
                dfs(node, [])
        return cycles

    all_cycles = find_cycles(all_graph)
    runtime_cycles = find_cycles(runtime_graph)
    runtime_keys = {"->".join(sorted(c)) for c in runtime_cycles}
    type_only = [c for c in all_cycles if "->".join(sorted(c)) not in runtime_keys]
    return {"all": all_cycles, "runtime": runtime_cycles, "typeOnly": type_only}


def is_entry_point(f: ParsedFile, script_entry_points: Set[str]) -> bool:
    return (
        f.is_executable
        or f.name == "__init__"
        or f.name == "setup"
        or f.name == "conftest"
        or f.path.startswith("scripts/")
        or f.path.startswith("examples/")
        or f.path in script_entry_points
    )


def detect_unused(
    files: List[ParsedFile],
    test_files: List[ParsedFile],
    script_entry_points: Set[str],
) -> Dict[str, list]:
    file_paths = {f.path for f in files}
    imported_files: Set[str] = set()
    imported_symbols: Dict[str, Set[str]] = {}
    for f in list(files) + list(test_files):
        for d in f.internal_dependencies:
            if d.file and d.file in file_paths:
                imported_files.add(d.file)
                syms = imported_symbols.setdefault(d.file, set())
                for imp in d.imports:
                    syms.add("*" if imp in ("*", "(module)") else imp)

    referenced_by_name: Set[str] = set()
    for f in files:
        for ref in f.referenced_file_names:
            referenced_by_name.add(ref)

    unused_files: List[str] = []
    for f in files:
        if is_entry_point(f, script_entry_points):
            continue
        if f"{f.name}.py" in referenced_by_name:
            continue
        if f.path not in imported_files:
            unused_files.append(f.path)

    unused_exports: List[dict] = []
    for f in files:
        used = imported_symbols.get(f.path)
        if not used or "*" in used:
            continue
        for fn in f.exports.functions:
            if fn not in used:
                unused_exports.append({"file": f.path, "name": fn, "type": "function"})
        for cls in f.exports.classes:
            if cls not in used:
                unused_exports.append({"file": f.path, "name": cls, "type": "class"})
        for iface in f.exports.interfaces:
            if iface not in used:
                unused_exports.append({"file": f.path, "name": iface, "type": "interface"})
        for en in f.exports.enums:
            if en not in used:
                unused_exports.append({"file": f.path, "name": en, "type": "enum"})
        for c in f.exports.constants:
            if c not in used:
                unused_exports.append({"file": f.path, "name": c, "type": "constant"})
    return {"unusedFiles": unused_files, "unusedExports": unused_exports}


def analyze_test_coverage(
    files: List[ParsedFile], test_files: List[ParsedFile]
) -> Dict[str, object]:
    """Map source files to the test files that import them (directly or via a barrel)."""
    file_paths = {f.path for f in files}
    by_path = {f.path: f for f in files}
    coverage: Dict[str, List[str]] = {f.path: [] for f in files}
    test_to_source: Dict[str, List[str]] = {}

    # Expand a barrel (__init__.py) import to the source files it re-exports from.
    def expand(path: str, seen: Set[str]) -> Set[str]:
        out = {path}
        f = by_path.get(path)
        if f and f.name == "__init__":
            for d in f.internal_dependencies:
                if d.file and d.file in file_paths and d.file not in seen:
                    seen.add(d.file)
                    out |= expand(d.file, seen)
        return out

    for tf in test_files:
        imported: List[str] = []
        for d in tf.internal_dependencies:
            if d.file and d.file in file_paths:
                for src in expand(d.file, set()):
                    if src in coverage and tf.path not in coverage[src]:
                        coverage[src].append(tf.path)
                    if src not in imported:
                        imported.append(src)
        test_to_source[tf.path] = imported

    tested = [p for p, t in coverage.items() if t]
    untested = [p for p, t in coverage.items() if not t]
    # Sort every list value so the serialized reports are deterministic across
    # runs (the per-test source lists are built from set iteration, whose order
    # is not stable -- otherwise regenerating churns test-coverage.json).
    return {
        "sourceFiles": [f.path for f in files],
        "testFiles": [f.path for f in test_files],
        "coverageMap": {k: sorted(v) for k, v in coverage.items()},
        "testToSourceMap": {k: sorted(v) for k, v in test_to_source.items()},
        "testedFiles": tested,
        "untestedFiles": untested,
    }


def generate_statistics(
    files: List[ParsedFile], modules: dict, circular: dict, unused: dict
) -> dict:
    return {
        "totalPythonFiles": len(files),
        "totalModules": len(modules),
        "totalLinesOfCode": sum(f.lines for f in files),
        "totalExports": sum(len(f.exports.named) for f in files),
        "totalClasses": sum(len(f.exports.classes) for f in files),
        "totalInterfaces": sum(len(f.exports.interfaces) for f in files),
        "totalEnums": sum(len(f.exports.enums) for f in files),
        "totalFunctions": sum(len(f.exports.functions) for f in files),
        "totalTypeGuards": sum(
            len([fn for fn in f.exports.functions if fn.startswith("is_")]) for f in files
        ),
        "totalConstants": sum(len(f.exports.constants) for f in files),
        "totalReExports": sum(len(f.exports.re_exported) for f in files),
        "totalTypeCheckingImports": sum(
            len([d for d in f.internal_dependencies if d.type_only]) for f in files
        ),
        "runtimeCircularDeps": len(circular["runtime"]),
        "typeCheckingCircularDeps": len(circular["typeOnly"]),
        "unusedFilesCount": len(unused["unusedFiles"]),
        "unusedExportsCount": len(unused["unusedExports"]),
    }


# --------------------------------------------------------------------------- #
# Discovery + project metadata.
# --------------------------------------------------------------------------- #
def _is_test_file(rel: str) -> bool:
    base = os.path.basename(rel)
    return (
        base.startswith("test_")
        or base.endswith("_test.py")
        or rel.startswith("tests/")
        or "/tests/" in f"/{rel}"
        or base == "conftest.py"
    )


def find_py_files(root: str, exclude: Set[str]) -> Tuple[List[str], List[str]]:
    """Return (source_abs_paths, test_abs_paths) for first-party .py files."""
    sources: List[str] = []
    tests: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            ap = os.path.join(dirpath, fn)
            rel = os.path.relpath(ap, root).replace(os.sep, "/")
            (tests if _is_test_file(rel) else sources).append(ap)
    return sorted(sources), sorted(tests)


def read_project_meta(root: str) -> Tuple[str, str, Set[str]]:
    """Return (name, version, script_entry_point_paths)."""
    name, version = "unknown", "0.0.0"
    entry_points: Set[str] = set()
    setup_py = os.path.join(root, "setup.py")
    if os.path.exists(setup_py):
        txt = open(setup_py, "r", encoding="utf-8", errors="replace").read()
        nm = re.search(r"""name\s*=\s*['"]([^'"]+)['"]""", txt)
        vm = re.search(r"""version\s*=\s*['"]([^'"]+)['"]""", txt)
        if nm:
            name = nm.group(1)
        if vm:
            version = vm.group(1)
        for m in re.finditer(r"""['"][\w.-]+\s*=\s*([\w.]+):[\w.]+['"]""", txt):
            entry_points.add(m.group(1).replace(".", "/") + ".py")
    pyproject = os.path.join(root, "pyproject.toml")
    if os.path.exists(pyproject):
        txt = open(pyproject, "r", encoding="utf-8", errors="replace").read()
        if name == "unknown":
            nm = re.search(r"""(?m)^\s*name\s*=\s*['"]([^'"]+)['"]""", txt)
            if nm:
                name = nm.group(1)
        if version == "0.0.0":
            vm = re.search(r"""(?m)^\s*version\s*=\s*['"]([^'"]+)['"]""", txt)
            if vm:
                version = vm.group(1)
    return name, version, entry_points


# --------------------------------------------------------------------------- #
# Output generation.
# --------------------------------------------------------------------------- #
def _title(name: str) -> str:
    return name[:1].upper() + name[1:].replace("-", " ").replace("/", " / ")


def file_to_json(f: ParsedFile) -> dict:
    data: Dict[str, object] = {
        "description": f.description or f"{f.name} module",
        "externalDependencies": [
            {"package": d.package, "imports": d.imports} for d in f.external_dependencies
        ],
        "stdlibDependencies": [
            {"module": d.package, "imports": d.imports} for d in f.stdlib_dependencies
        ],
        "internalDependencies": [
            {
                "file": d.file,
                "module": d.module,
                "imports": d.imports,
                **({"reExport": True} if d.re_export else {}),
                **({"typeOnly": True} if d.type_only else {}),
            }
            for d in f.internal_dependencies
        ],
        "exports": f.exports.named,
    }
    if f.exports.re_exported:
        data["reExported"] = f.exports.re_exported
    if f.exports.classes:
        data["classes"] = f.exports.classes
    if f.exports.interfaces:
        data["interfaces"] = f.exports.interfaces
    if f.exports.enums:
        data["enums"] = f.exports.enums
    if f.exports.functions:
        data["functions"] = f.exports.functions
    if f.exports.constants:
        data["constants"] = f.exports.constants
    return data


def generate_json(files, modules, stats, circular, name, version, script_eps) -> dict:
    today = date.today().isoformat()
    modules_json: Dict[str, Dict[str, dict]] = {}
    for cat, cat_files in modules.items():
        modules_json[cat] = {p: file_to_json(f) for p, f in cat_files.items()}
    layers = [
        {"name": _title(n), "files": list(modules[n].keys())} for n in modules if modules[n]
    ]
    entry_points = [
        {
            "file": f.path,
            "type": "cli" if (f.is_executable or f.path.startswith("scripts/")) else "main",
            "description": f.description or "Entry Point",
        }
        for f in files
        if is_entry_point(f, script_eps)
    ]
    return {
        "metadata": {
            "name": name,
            "version": version,
            "lastUpdated": today,
            "totalFiles": stats["totalPythonFiles"],
            "totalModules": stats["totalModules"],
            "totalExports": stats["totalExports"],
        },
        "entryPoints": entry_points,
        "modules": modules_json,
        "dependencyGraph": {
            "circularDependencies": {
                "runtime": circular["runtime"],
                "typeOnly": circular["typeOnly"],
                "total": len(circular["all"]),
                "runtimeCount": len(circular["runtime"]),
                "typeOnlyCount": len(circular["typeOnly"]),
            },
            "layers": layers,
        },
        "statistics": stats,
    }


def generate_mermaid(modules, files) -> str:
    lines = ["```mermaid", "graph TD"]
    node_ids: Dict[str, str] = {}
    counter = 0
    for mod_name, mod_files in modules.items():
        lines.append(f"    subgraph {_title(mod_name)}")
        paths = list(mod_files.keys())
        for p in paths[:5]:
            nid = f"N{counter}"
            counter += 1
            node_ids[p] = nid
            lines.append(f"        {nid}[{os.path.splitext(os.path.basename(p))[0]}]")
        if len(paths) > 5:
            lines.append(f"        N{counter}[...{len(paths) - 5} more]")
            counter += 1
        lines.append("    end")
        lines.append("")
    added: Set[str] = set()
    edges = 0
    for f in files:
        src = node_ids.get(f.path)
        if not src:
            continue
        for d in f.internal_dependencies:
            if edges >= 30:
                break
            tgt = node_ids.get(d.file) if d.file else None
            if tgt and src != tgt and f"{src}-{tgt}" not in added:
                lines.append(f"    {src} --> {tgt}")
                added.add(f"{src}-{tgt}")
                edges += 1
    lines.append("```")
    return "\n".join(lines)


def generate_markdown(files, modules, stats, circular, name, version) -> str:
    today = date.today().isoformat()
    L: List[str] = []
    L.append(f"# {name} - Dependency Graph")
    L.append("")
    L.append(f"**Version**: {version} | **Last Updated**: {today}")
    L.append("")
    L.append(
        "Comprehensive dependency graph of all Python modules, imports, exports, "
        "functions, classes, and constants in the codebase."
    )
    L.append("")
    L.append("---")
    L.append("")
    L.append("## Overview")
    L.append("")
    L.append("The codebase is organized into the following modules:")
    L.append("")
    for mod_name, mod_files in modules.items():
        c = len(mod_files)
        L.append(f"- **{mod_name}**: {c} file{'s' if c != 1 else ''}")
    L.append("")
    L.append("---")
    L.append("")
    for cat, cat_files in modules.items():
        L.append(f"## {_title(cat)} Dependencies")
        L.append("")
        for path, f in cat_files.items():
            L.append(f"### `{path}` - {f.description or f.name + ' module'}")
            L.append("")
            if f.external_dependencies:
                L.append("**Third-party Dependencies:**")
                L.append("| Package | Import |")
                L.append("|---------|--------|")
                for d in f.external_dependencies:
                    L.append(f"| `{d.package}` | `{', '.join(d.imports)}` |")
                L.append("")
            if f.stdlib_dependencies:
                L.append("**Standard-library Dependencies:**")
                L.append("| Module | Import |")
                L.append("|--------|--------|")
                for d in f.stdlib_dependencies:
                    L.append(f"| `{d.package}` | `{', '.join(d.imports)}` |")
                L.append("")
            if f.internal_dependencies:
                L.append("**Internal Dependencies:**")
                L.append("| Module | Imports | Type |")
                L.append("|--------|---------|------|")
                for d in f.internal_dependencies:
                    usage = "Re-export" if d.re_export else "Import"
                    if d.type_only:
                        usage += " (TYPE_CHECKING)"
                    L.append(f"| `{d.file or d.module}` | `{', '.join(d.imports)}` | {usage} |")
                L.append("")
            ex = f.exports
            if ex.named or ex.re_exported:
                L.append("**Exports:**")
                if ex.classes:
                    L.append(f"- Classes: `{'`, `'.join(ex.classes)}`")
                if ex.interfaces:
                    L.append(f"- Protocols/ABCs: `{'`, `'.join(ex.interfaces)}`")
                if ex.enums:
                    L.append(f"- Enums: `{'`, `'.join(ex.enums)}`")
                if ex.functions:
                    L.append(f"- Functions: `{'`, `'.join(ex.functions)}`")
                if ex.constants:
                    L.append(f"- Constants: `{'`, `'.join(ex.constants)}`")
                if ex.re_exported:
                    L.append(f"- Re-exports: `{'`, `'.join(ex.re_exported)}`")
                L.append("")
            L.append("---")
            L.append("")
    L.append("## Circular Dependency Analysis")
    L.append("")
    if not circular["all"]:
        L.append("**No circular dependencies detected.**")
    else:
        L.append(f"**{len(circular['all'])} circular dependencies detected:**")
        L.append("")
        L.append(f"- **Runtime cycles**: {len(circular['runtime'])} (require attention)")
        L.append(f"- **TYPE_CHECKING-only cycles**: {len(circular['typeOnly'])} (safe)")
        L.append("")
        if circular["runtime"]:
            L.append("### Runtime Circular Dependencies")
            L.append("")
            for c in circular["runtime"][:10]:
                L.append(f"- {' -> '.join(c)}")
            L.append("")
        if circular["typeOnly"]:
            L.append("### TYPE_CHECKING-only Circular Dependencies")
            L.append("")
            for c in circular["typeOnly"][:10]:
                L.append(f"- {' -> '.join(c)}")
            L.append("")
    L.append("---")
    L.append("")
    L.append("## Visual Dependency Graph")
    L.append("")
    L.append(generate_mermaid(modules, files))
    L.append("")
    L.append("---")
    L.append("")
    L.append("## Summary Statistics")
    L.append("")
    L.append("| Category | Count |")
    L.append("|----------|-------|")
    labels = {
        "totalPythonFiles": "Total Python Files",
        "totalModules": "Total Modules",
        "totalLinesOfCode": "Total Lines of Code",
        "totalExports": "Total Public Exports",
        "totalReExports": "Total Re-exports",
        "totalClasses": "Total Classes",
        "totalInterfaces": "Total Protocols/ABCs",
        "totalEnums": "Total Enums",
        "totalFunctions": "Total Functions",
        "totalTypeGuards": "Total Type Guards (is_*)",
        "totalConstants": "Total Constants",
        "totalTypeCheckingImports": "TYPE_CHECKING Imports",
        "runtimeCircularDeps": "Runtime Circular Deps",
        "typeCheckingCircularDeps": "TYPE_CHECKING Circular Deps",
        "unusedFilesCount": "Potentially Unused Files",
        "unusedExportsCount": "Potentially Unused Exports",
    }
    for key, lab in labels.items():
        L.append(f"| {lab} | {stats[key]} |")
    L.append("")
    L.append(f"*Last Updated*: {today}  |  *Version*: {version}")
    L.append("")
    return "\n".join(L)


def generate_compact(files, modules, stats, circular, name, version) -> str:
    summary: Dict[str, object] = {
        "m": {
            "n": name, "v": version, "d": date.today().isoformat(),
            "f": stats["totalPythonFiles"], "e": stats["totalExports"],
            "re": stats["totalReExports"],
        },
        "s": {
            "loc": stats["totalLinesOfCode"], "cls": stats["totalClasses"],
            "int": stats["totalInterfaces"], "fn": stats["totalFunctions"],
            "en": stats["totalEnums"], "co": stats["totalConstants"],
            "tci": stats["totalTypeCheckingImports"],
        },
        "c": {
            "rt": len(circular["runtime"]), "to": len(circular["typeOnly"]),
            "rtp": [
                "->".join(os.path.splitext(os.path.basename(p))[0] for p in c)
                for c in circular["runtime"][:5]
            ],
        },
        "mod": {},
        "hp": [],
    }
    mod_summary: Dict[str, dict] = {}
    for mod_name, mod_files in modules.items():
        flist = list(mod_files.values())
        exports: List[str] = []
        for f in flist:
            exports.extend(f.exports.named)
        entry: Dict[str, object] = {"f": len(mod_files), "exp": list(dict.fromkeys(exports))[:20]}
        classes: List[str] = []
        for f in flist:
            classes.extend(f.exports.classes)
        if classes:
            entry["cls"] = list(dict.fromkeys(classes))
        mod_summary[mod_name] = entry
    summary["mod"] = mod_summary
    conn: List[dict] = []
    for f in files:
        out_count = sum(
            1 for o in files if any(d.file == f.path for d in o.internal_dependencies)
        )
        conn.append(
            {
                "p": "/".join(f.path.split("/")[-2:]),
                "i": len(f.internal_dependencies),
                "o": out_count,
            }
        )
    conn.sort(key=lambda x: x["i"] + x["o"], reverse=True)
    summary["hp"] = conn[:15]
    return json.dumps(summary, separators=(",", ":"))


def generate_test_coverage_md(cov: dict) -> str:
    today = date.today().isoformat()
    L: List[str] = ["# Test Coverage Analysis", "", f"**Generated**: {today}", ""]
    total = len(cov["sourceFiles"])
    tested = len(cov["testedFiles"])
    pct = f"{(tested / total * 100):.1f}" if total else "0"
    L += [
        "## Summary", "", "| Metric | Count |", "|--------|-------|",
        f"| Total Source Files | {total} |",
        f"| Total Test Files | {len(cov['testFiles'])} |",
        f"| Source Files with Tests | {tested} |",
        f"| Source Files without Tests | {len(cov['untestedFiles'])} |",
        f"| Coverage | {pct}% |", "", "---", "",
        "## Source Files Without Test Coverage", "",
    ]
    if not cov["untestedFiles"]:
        L.append("**All source files have test coverage!**")
    else:
        L.append(f"{len(cov['untestedFiles'])} source files are not imported by any test:")
        L.append("")
        for f in sorted(cov["untestedFiles"]):
            L.append(f"- `{f}`")
    L += ["", "---", "", "## Source Files With Test Coverage", "",
          "| Source File | Test Files |", "|-------------|------------|"]
    for src in sorted(cov["testedFiles"]):
        tests = cov["coverageMap"].get(src, [])
        short = ", ".join(f"`{os.path.basename(t)}`" for t in tests)
        L.append(f"| `{src}` | {short} |")
    L.append("")
    return "\n".join(L)


def generate_test_coverage_json(cov: dict) -> dict:
    total = len(cov["sourceFiles"])
    tested = len(cov["testedFiles"])
    return {
        "metadata": {
            "generatedAt": date.today().isoformat(),
            "totalSourceFiles": total,
            "totalTestFiles": len(cov["testFiles"]),
            "testedCount": tested,
            "untestedCount": len(cov["untestedFiles"]),
            "coveragePercent": f"{(tested / total * 100):.1f}" if total else "0",
        },
        "untestedFiles": sorted(cov["untestedFiles"]),
        "testedFiles": sorted(cov["testedFiles"]),
        "coverageMap": cov["coverageMap"],
        "testToSourceMap": cov["testToSourceMap"],
    }


def generate_unused_md(unused: dict) -> str:
    today = date.today().isoformat()
    L = [
        "# Unused Files and Exports Analysis", "", f"**Generated**: {today}", "",
        "## Summary", "",
        f"- **Potentially unused files**: {len(unused['unusedFiles'])}",
        f"- **Potentially unused exports**: {len(unused['unusedExports'])}", "",
        "## Potentially Unused Files", "",
    ]
    if not unused["unusedFiles"]:
        L.append("None.")
    else:
        for f in unused["unusedFiles"]:
            L.append(f"- `{f}`")
    L += ["", "## Potentially Unused Exports", ""]
    by_file: Dict[str, List[dict]] = {}
    for e in unused["unusedExports"]:
        by_file.setdefault(e["file"], []).append(e)
    if not by_file:
        L.append("None.")
    for f, exps in by_file.items():
        L.append(f"### `{f}`")
        L.append("")
        for e in exps:
            L.append(f"- `{e['name']}` ({e['type']})")
        L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# CLI / main.
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a Python dependency graph under docs/architecture/.",
    )
    p.add_argument("root", nargs="?", default=os.getcwd(),
                   help="Project root directory (default: current directory)")
    p.add_argument("--root", dest="root_opt", default=None,
                   help="Project root directory (overrides positional)")
    p.add_argument("--exclude", default=None,
                   help="Replace the default skip list (comma-separated dir names)")
    p.add_argument("--also-exclude", default=None,
                   help="Add directory names to the default skip list")
    p.add_argument("--include-tests", "-t", action="store_true",
                   help="Include test files in dependency / coverage analysis")
    return p


def run(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = os.path.abspath(args.root_opt or args.root)
    if args.exclude is not None:
        exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}
    else:
        exclude = set(DEFAULT_EXCLUDE_DIRS)
    if args.also_exclude:
        exclude |= {s.strip() for s in args.also_exclude.split(",") if s.strip()}

    output_dir = os.path.join(root, "docs", "architecture")
    os.makedirs(output_dir, exist_ok=True)

    name, version, script_eps = read_project_meta(root)
    source_abs, test_abs = find_py_files(root, exclude)
    all_rel = {
        os.path.relpath(p, root).replace(os.sep, "/") for p in source_abs + test_abs
    }
    package_roots = discover_package_roots(root, exclude)

    print(f"Scanning {root}")
    print(f"Skipping: {', '.join(sorted(exclude))}")
    print(f"Package roots: {package_roots}")
    print(f"Found {len(source_abs)} source files, {len(test_abs)} test files")

    files = [parse_file(p, root, package_roots, all_rel) for p in source_abs]
    test_files = (
        [parse_file(p, root, package_roots, all_rel) for p in test_abs]
        if args.include_tests
        else []
    )

    modules = categorize_files(files)
    circular = detect_circular(files)
    unused = detect_unused(files, test_files, script_eps)
    stats = generate_statistics(files, modules, circular, unused)

    json_obj = generate_json(files, modules, stats, circular, name, version, script_eps)
    with open(os.path.join(output_dir, "dependency-graph.json"), "w", encoding="utf-8") as fh:
        json.dump(json_obj, fh, indent=2)
    print("Written: docs/architecture/dependency-graph.json")

    try:
        import yaml  # type: ignore
        with open(os.path.join(output_dir, "dependency-graph.yaml"), "w", encoding="utf-8") as fh:
            yaml.dump(json_obj, fh, indent=2, sort_keys=False, allow_unicode=True, width=120)
        print("Written: docs/architecture/dependency-graph.yaml")
    except ImportError:
        print("PyYAML not installed; skipping dependency-graph.yaml")

    with open(os.path.join(output_dir, "DEPENDENCY_GRAPH.md"), "w", encoding="utf-8") as fh:
        fh.write(generate_markdown(files, modules, stats, circular, name, version))
    print("Written: docs/architecture/DEPENDENCY_GRAPH.md")

    compact_path = os.path.join(output_dir, "dependency-summary.compact.json")
    with open(compact_path, "w", encoding="utf-8") as fh:
        fh.write(generate_compact(files, modules, stats, circular, name, version))
    print("Written: docs/architecture/dependency-summary.compact.json")

    with open(os.path.join(output_dir, "unused-analysis.md"), "w", encoding="utf-8") as fh:
        fh.write(generate_unused_md(unused))
    print("Written: docs/architecture/unused-analysis.md")

    if args.include_tests:
        cov = analyze_test_coverage(files, test_files)
        with open(os.path.join(output_dir, "TEST_COVERAGE.md"), "w", encoding="utf-8") as fh:
            fh.write(generate_test_coverage_md(cov))
        with open(os.path.join(output_dir, "test-coverage.json"), "w", encoding="utf-8") as fh:
            json.dump(generate_test_coverage_json(cov), fh, indent=2)
        print("Written: docs/architecture/TEST_COVERAGE.md + test-coverage.json")

    print("\nDependency graph generation complete.")
    print(
        f"  - {stats['totalPythonFiles']} files, {stats['totalModules']} modules, "
        f"{stats['totalLinesOfCode']} LOC"
    )
    print(f"  - {stats['totalExports']} exports ({stats['totalReExports']} re-exports)")
    print(
        f"  - {stats['runtimeCircularDeps']} runtime circular deps, "
        f"{stats['typeCheckingCircularDeps']} TYPE_CHECKING-only"
    )
    print(
        f"  - {stats['unusedFilesCount']} potentially unused files, "
        f"{stats['unusedExportsCount']} potentially unused exports"
    )
    if unused["unusedFiles"]:
        print("\nPotentially unused files:")
        for f in unused["unusedFiles"][:20]:
            print(f"  - {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
