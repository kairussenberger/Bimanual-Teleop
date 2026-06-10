from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_guard():
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_no_mujoco_runtime.py"
    spec = importlib.util.spec_from_file_location("check_no_mujoco_runtime", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_mujoco_guard_detects_forbidden_import_roots(tmp_path):
    guard = _load_guard()
    module = tmp_path / "bad_runtime.py"
    module.write_text(
        "import mujoco.viewer\n"
        "from dm_control import mjcf\n"
        "from mink.tasks import FrameTask\n",
        encoding="utf-8",
    )

    assert {"mujoco", "dm_control", "mink"} <= guard._import_roots(module)


def test_no_mujoco_guard_parses_project_dependencies(tmp_path, monkeypatch):
    guard = _load_guard()
    (tmp_path / "pyproject.toml").write_text(
        "[project]\n"
        'dependencies = ["numpy>=2", "mujoco==3.2.0"]\n'
        "[project.optional-dependencies]\n"
        'dev = ["dm-control>=1"]\n'
        "[dependency-groups]\n"
        'test = ["mink>=0.0.6"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(guard, "REPO", tmp_path)

    assert {"mujoco", "dm-control", "mink"} <= guard._dependency_names()


def test_no_mujoco_guard_parses_uv_lock_packages(tmp_path, monkeypatch):
    guard = _load_guard()
    (tmp_path / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "mujoco"\n\n[[package]]\nname = "numpy"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(guard, "REPO", tmp_path)

    assert guard._locked_package_names() == {"mujoco", "numpy"}


def test_no_mujoco_guard_main_rejects_forbidden_lockfile_entry(tmp_path, monkeypatch):
    guard = _load_guard()
    src = tmp_path / "src"
    scripts = tmp_path / "scripts"
    src.mkdir()
    scripts.mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\ndependencies = []\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text('version = 1\n\n[[package]]\nname = "mink"\n', encoding="utf-8")

    monkeypatch.setattr(guard, "REPO", tmp_path)
    monkeypatch.setattr(guard, "PY_SCAN_ROOTS", [src, scripts])

    with pytest.raises(AssertionError, match="uv.lock contains mink"):
        guard.main()
