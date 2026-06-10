#!/usr/bin/env python
"""Fail if MuJoCo/mink runtime dependencies or imports reappear.

Historical docs and the source MJCF assets are allowed. Runtime Python modules,
launchers, scripts, and project dependencies must stay MuJoCo-free.
"""
from __future__ import annotations

import ast
import tomllib
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
FORBIDDEN_ROOTS = {"mujoco", "mink", "dm_control"}
FORBIDDEN_PACKAGES = {"mujoco", "mink", "dm-control", "dm_control"}
PY_SCAN_ROOTS = [
    REPO / "src" / "bimanual_teleop",
    REPO / "scripts",
]
SKIP_PARTS = {
    "__pycache__",
    "sim/models",
}


def _skip(path: Path) -> bool:
    rel = path.relative_to(REPO).as_posix()
    return any(part in rel for part in SKIP_PARTS)


def _import_roots(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        raise AssertionError(f"{path.relative_to(REPO)} does not parse: {exc}") from exc
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for root in PY_SCAN_ROOTS:
        files.extend(p for p in root.rglob("*.py") if not _skip(p))
    return sorted(files)


def _dependency_names() -> set[str]:
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    names: set[str] = set()
    for dep in data.get("project", {}).get("dependencies", []):
        names.add(str(dep).split("[", 1)[0].split(">", 1)[0].split("=", 1)[0].split("<", 1)[0].strip().lower())
    for deps in data.get("project", {}).get("optional-dependencies", {}).values():
        for dep in deps:
            names.add(str(dep).split("[", 1)[0].split(">", 1)[0].split("=", 1)[0].split("<", 1)[0].strip().lower())
    for deps in data.get("dependency-groups", {}).values():
        for dep in deps:
            names.add(str(dep).split("[", 1)[0].split(">", 1)[0].split("=", 1)[0].split("<", 1)[0].strip().lower())
    return names


def _locked_package_names() -> set[str]:
    lock = REPO / "uv.lock"
    if not lock.exists():
        return set()
    data = tomllib.loads(lock.read_text(encoding="utf-8"))
    return {str(pkg.get("name", "")).lower() for pkg in data.get("package", [])}


def main() -> int:
    offenders: list[str] = []
    for path in _iter_python_files():
        bad = _import_roots(path) & FORBIDDEN_ROOTS
        if bad:
            offenders.append(f"{path.relative_to(REPO)} imports {', '.join(sorted(bad))}")

    deps = _dependency_names()
    bad_deps = deps & FORBIDDEN_PACKAGES
    if bad_deps:
        offenders.append(f"pyproject.toml depends on {', '.join(sorted(bad_deps))}")
    locked = _locked_package_names()
    bad_locked = locked & FORBIDDEN_PACKAGES
    if bad_locked:
        offenders.append(f"uv.lock contains {', '.join(sorted(bad_locked))}")

    if offenders:
        raise AssertionError("MuJoCo/mink runtime dependency found:\n" + "\n".join(offenders))

    print("runtime imports and dependencies are MuJoCo-free")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
