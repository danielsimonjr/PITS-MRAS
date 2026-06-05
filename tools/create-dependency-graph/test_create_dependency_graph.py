"""Tests for the standalone Python dependency-graph tool.

Run with:  pytest tools/create-dependency-graph/

(The tool lives in a hyphenated directory and is not an importable package, so we
load it from its file path via importlib.)
"""

import importlib.util
import os
import sys
import textwrap

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "cdg", os.path.join(_HERE, "create_dependency_graph.py")
)
cdg = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
# Register before exec so dataclass field resolution (with `from __future__
# import annotations`) can find the module's namespace.
sys.modules["cdg"] = cdg
_spec.loader.exec_module(cdg)


# --------------------------------------------------------------------------- #
# Import-statement parsing.
# --------------------------------------------------------------------------- #
def test_split_import_names_handles_aliases_and_star():
    assert cdg._split_import_names("a, b as c, d") == ["a", "c", "d"]
    assert cdg._split_import_names("*") == ["*"]


def test_logical_import_lines_joins_parenthesized():
    content = textwrap.dedent(
        """\
        from pkg.mod import (
            Alpha,
            Beta,
        )
        x = 1
        """
    )
    lines = cdg._logical_import_lines(content)
    assert len(lines) == 1
    start, joined = lines[0]
    assert start == 0
    assert "Alpha" in joined and "Beta" in joined and ")" in joined


def test_typechecking_ranges_detected():
    content = textwrap.dedent(
        """\
        from __future__ import annotations
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            from .other import Thing

        import os
        """
    )
    ranges = cdg._typechecking_line_ranges(content)
    # The `from .other import Thing` line (index 4) is guarded.
    assert any(start <= 4 < end for start, end in ranges)
    # `import os` (index 6) is not.
    assert not any(start <= 6 < end for start, end in ranges)


# --------------------------------------------------------------------------- #
# Relative-import resolution.
# --------------------------------------------------------------------------- #
def test_resolve_relative_one_and_two_dots():
    # from .mras import X inside src/pkg/controllers/__init__.py
    assert (
        cdg.resolve_relative("src/pkg/controllers/__init__.py", 1, "mras")
        == "src/pkg/controllers/mras"
    )
    # from ..models.critic import Y inside src/pkg/controllers/mras.py
    assert (
        cdg.resolve_relative("src/pkg/controllers/mras.py", 2, "models.critic")
        == "src/pkg/models/critic"
    )


# --------------------------------------------------------------------------- #
# Full parse on temp files.
# --------------------------------------------------------------------------- #
def _write(tmp_path, rel, body):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(p)


def test_parse_classifies_imports_and_exports(tmp_path):
    root = str(tmp_path)
    _write(tmp_path, "src/pkg/__init__.py", "")
    _write(tmp_path, "src/pkg/models/__init__.py", "")
    target = _write(
        tmp_path,
        "src/pkg/models/critic.py",
        '''\
        """The critic module docstring."""
        from __future__ import annotations
        import torch
        import os
        from ..config import Config
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from .decoders import Decoder

        CONST_VALUE = 3

        class QuadraticCritic:
            pass

        def helper():
            return 1

        def _private():
            return 0
        ''',
    )
    _write(tmp_path, "src/pkg/config.py", "class Config: pass\n")
    _write(tmp_path, "src/pkg/models/decoders.py", "class Decoder: pass\n")

    all_rel = {
        "src/pkg/__init__.py", "src/pkg/models/__init__.py",
        "src/pkg/models/critic.py", "src/pkg/config.py", "src/pkg/models/decoders.py",
    }
    roots = cdg.discover_package_roots(root, set(cdg.DEFAULT_EXCLUDE_DIRS))
    assert roots.get("pkg") == "src/pkg"

    pf = cdg.parse_file(target, root, roots, all_rel)
    assert pf.description == "The critic module docstring."
    # torch is third-party; os is stdlib.
    assert any(d.package == "torch" for d in pf.external_dependencies)
    assert any(d.package == "os" for d in pf.stdlib_dependencies)
    # `from ..config import Config` resolves to the config file (runtime).
    cfg = [d for d in pf.internal_dependencies if d.file == "src/pkg/config.py"]
    assert cfg and not cfg[0].type_only
    # TYPE_CHECKING import of Decoder is internal AND type-only.
    dec = [d for d in pf.internal_dependencies if d.file == "src/pkg/models/decoders.py"]
    assert dec and dec[0].type_only
    # Exports: public class + function + constant; private excluded.
    assert "QuadraticCritic" in pf.exports.classes
    assert "helper" in pf.exports.functions
    assert "_private" not in pf.exports.named
    assert "CONST_VALUE" in pf.exports.constants


def test_init_barrel_reexports(tmp_path):
    root = str(tmp_path)
    _write(tmp_path, "src/pkg/__init__.py",
           "from .models.critic import QuadraticCritic\n__all__ = ['QuadraticCritic']\n")
    _write(tmp_path, "src/pkg/models/__init__.py", "")
    _write(tmp_path, "src/pkg/models/critic.py", "class QuadraticCritic: pass\n")
    all_rel = {"src/pkg/__init__.py", "src/pkg/models/__init__.py", "src/pkg/models/critic.py"}
    roots = cdg.discover_package_roots(root, set(cdg.DEFAULT_EXCLUDE_DIRS))
    pf = cdg.parse_file(str(tmp_path / "src/pkg/__init__.py"), root, roots, all_rel)
    dep = [d for d in pf.internal_dependencies if d.file == "src/pkg/models/critic.py"]
    assert dep and dep[0].re_export
    assert "QuadraticCritic" in pf.exports.re_exported


def test_init_barrel_reexports_absolute_imports(tmp_path):
    """An ABSOLUTE intra-package from-import in __init__.py is also a re-export."""
    root = str(tmp_path)
    _write(tmp_path, "src/pkg/__init__.py",
           "from pkg.models.critic import QuadraticCritic\n__all__ = ['QuadraticCritic']\n")
    _write(tmp_path, "src/pkg/models/__init__.py", "")
    _write(tmp_path, "src/pkg/models/critic.py", "class QuadraticCritic: pass\n")
    all_rel = {"src/pkg/__init__.py", "src/pkg/models/__init__.py", "src/pkg/models/critic.py"}
    roots = cdg.discover_package_roots(root, set(cdg.DEFAULT_EXCLUDE_DIRS))
    pf = cdg.parse_file(str(tmp_path / "src/pkg/__init__.py"), root, roots, all_rel)
    dep = [d for d in pf.internal_dependencies if d.file == "src/pkg/models/critic.py"]
    assert dep and dep[0].re_export
    assert "QuadraticCritic" in pf.exports.re_exported


# --------------------------------------------------------------------------- #
# Analysis: circular deps, unused, coverage.
# --------------------------------------------------------------------------- #
def _pf(path, deps, type_only_to=()):
    f = cdg.ParsedFile(path=path, name=os.path.splitext(os.path.basename(path))[0])
    for target in deps:
        f.internal_dependencies.append(
            cdg.InternalDep(file=target, module=target, imports=["X"],
                            type_only=target in type_only_to)
        )
    return f


def test_detect_circular_runtime_vs_typechecking():
    a = _pf("a.py", ["b.py"])
    b = _pf("b.py", ["a.py"], type_only_to=("a.py",))  # b -> a is TYPE_CHECKING only
    res = cdg.detect_circular([a, b])
    assert len(res["all"]) == 1
    assert len(res["runtime"]) == 0  # the back-edge is type-only -> no runtime cycle
    assert len(res["typeOnly"]) == 1


def test_detect_unused_files_and_exports():
    used = cdg.ParsedFile(path="used.py", name="used")
    used.exports.functions = ["live", "dead"]
    used.exports.named = ["live", "dead"]
    orphan = cdg.ParsedFile(path="orphan.py", name="orphan")
    importer = _pf("importer.py", ["used.py"])
    importer.internal_dependencies[0].imports = ["live"]
    res = cdg.detect_unused([used, orphan, importer], [], set())
    assert "orphan.py" in res["unusedFiles"]
    names = {(e["file"], e["name"]) for e in res["unusedExports"]}
    assert ("used.py", "dead") in names
    assert ("used.py", "live") not in names


def test_analyze_test_coverage_direct_and_barrel():
    init = cdg.ParsedFile(path="src/pkg/__init__.py", name="__init__")
    init.internal_dependencies.append(
        cdg.InternalDep(file="src/pkg/core.py", module=".core", imports=["Core"], re_export=True)
    )
    core = cdg.ParsedFile(path="src/pkg/core.py", name="core")
    test = _pf("tests/test_core.py", ["src/pkg/__init__.py"])  # imports the barrel
    cov = cdg.analyze_test_coverage([init, core], [test])
    # Coverage traces through the barrel to core.py.
    assert "src/pkg/core.py" in cov["testedFiles"]


def test_analyze_test_coverage_maps_are_sorted_for_reproducibility():
    """coverageMap / testToSourceMap list values are sorted, so regenerating the
    reports is idempotent (no set-iteration ordering churn between runs)."""
    init = cdg.ParsedFile(path="src/pkg/__init__.py", name="__init__")
    mods = ("zeta", "alpha", "mid", "beta", "gamma")  # deliberately unsorted
    for m in mods:
        init.internal_dependencies.append(
            cdg.InternalDep(
                file=f"src/pkg/{m}.py", module=f".{m}", imports=[m], re_export=True
            )
        )
    srcs = [cdg.ParsedFile(path=f"src/pkg/{m}.py", name=m) for m in mods]
    test = _pf("tests/test_all.py", ["src/pkg/__init__.py"])  # imports the barrel

    cov = cdg.analyze_test_coverage([init, *srcs], [test])

    t2s = cov["testToSourceMap"]["tests/test_all.py"]
    assert t2s == sorted(t2s), "testToSourceMap values must be sorted (deterministic)"
    assert {f"src/pkg/{m}.py" for m in mods} <= set(t2s)
    for sources in cov["coverageMap"].values():
        assert sources == sorted(sources), "coverageMap values must be sorted"
