from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


SKIP_ALL_FLAGS = [
    "--skip-pytest",
    "--skip-rig-contract",
    "--skip-no-mujoco-runtime",
    "--skip-body-relative",
    "--skip-body-relative-render",
    "--skip-yam-geometry",
    "--skip-synthetic",
    "--skip-unity-contract",
    "--skip-cli",
    "--skip-smoke",
    "--skip-record-replay",
    "--skip-json-monitor",
]


def _load_verify_stack():
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_stack.py"
    spec = importlib.util.spec_from_file_location("verify_stack", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_verify_stack_skips_unity_editor_gate_by_default(monkeypatch):
    verifier = _load_verify_stack()
    calls: list[tuple[str, list[str]]] = []

    def fake_run_step(name: str, cmd: list[str]) -> None:
        calls.append((name, cmd))

    monkeypatch.setattr(verifier, "run_step", fake_run_step)
    monkeypatch.setattr(sys, "argv", ["verify_stack.py", *SKIP_ALL_FLAGS])

    assert verifier.main() == 0
    assert calls == []


def test_verify_stack_unity_editor_gate_is_required_when_requested(monkeypatch):
    verifier = _load_verify_stack()
    calls: list[tuple[str, list[str]]] = []

    def fake_run_step(name: str, cmd: list[str]) -> None:
        calls.append((name, cmd))

    monkeypatch.setattr(verifier, "run_step", fake_run_step)
    monkeypatch.setattr(sys, "argv", ["verify_stack.py", *SKIP_ALL_FLAGS, "--unity-editor"])

    assert verifier.main() == 0
    assert calls == [
        (
            "Unity Editor batch validation",
            ["uv", "run", "python", "scripts/run_unity_validation.py", "--require"],
        )
    ]


def test_verify_stack_returns_unity_editor_failure_code(monkeypatch):
    verifier = _load_verify_stack()

    def fake_run_step(name: str, cmd: list[str]) -> None:
        raise subprocess.CalledProcessError(2, cmd)

    monkeypatch.setattr(verifier, "run_step", fake_run_step)
    monkeypatch.setattr(sys, "argv", ["verify_stack.py", *SKIP_ALL_FLAGS, "--unity-editor"])

    assert verifier.main() == 2


def test_verify_stack_cli_smoke_includes_quest_diagnostic(monkeypatch):
    verifier = _load_verify_stack()
    calls: list[tuple[str, list[str]]] = []

    def fake_run_step(name: str, cmd: list[str]) -> None:
        calls.append((name, cmd))

    monkeypatch.setattr(verifier, "run_step", fake_run_step)

    verifier.run_launch_cli_smoke()

    assert calls == [
        (
            "launch CLI help",
            ["uv", "run", "python", "-m", "bimanual_teleop.launch.run_teleop", "--help"],
        ),
        (
            "hardware CLI help",
            ["uv", "run", "python", "-m", "bimanual_teleop.launch.run_hw", "--help"],
        ),
        (
            "Quest diagnostic CLI help",
            ["uv", "run", "python", "scripts/check_quest.py", "--help"],
        ),
        (
            "Quest roll diagnostic CLI help",
            ["uv", "run", "python", "scripts/check_roll.py", "--help"],
        ),
    ]


def test_verify_stack_json_monitor_smoke_requires_command_targets(monkeypatch):
    verifier = _load_verify_stack()
    popen_calls: list[list[str]] = []
    run_calls: list[list[str]] = []
    sleeps: list[float] = []

    class FakeProc:
        def __init__(self, cmd: list[str], cwd):
            popen_calls.append(cmd)
            self._done = False

        def wait(self, timeout=None):
            self._done = True
            return 0

        def poll(self):
            return 0 if self._done else None

        def terminate(self):
            self._done = True

        def kill(self):
            self._done = True

    def fake_run(cmd: list[str], cwd, check: bool):
        run_calls.append(cmd)
        assert check is True

    monkeypatch.setattr(verifier.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(verifier.subprocess, "run", fake_run)
    monkeypatch.setattr(verifier.time, "sleep", lambda value: sleeps.append(value))

    verifier.run_json_monitor_smoke(0.25)

    assert "--calib-seconds" in popen_calls[0]
    assert "0" == popen_calls[0][popen_calls[0].index("--calib-seconds") + 1]
    assert "--require-command-target" in run_calls[0]
    assert sleeps == [0.8]


def test_verify_stack_runs_body_relative_render_probe_by_default(monkeypatch):
    verifier = _load_verify_stack()
    calls: list[tuple[str, list[str]]] = []

    def fake_run_step(name: str, cmd: list[str]) -> None:
        calls.append((name, cmd))

    monkeypatch.setattr(verifier, "run_step", fake_run_step)
    monkeypatch.setattr(verifier, "run_launch_cli_smoke", lambda: None)
    monkeypatch.setattr(verifier, "run_record_replay_smoke", lambda seconds: None)
    monkeypatch.setattr(verifier, "run_json_monitor_smoke", lambda seconds: None)
    flags = [
        flag for flag in SKIP_ALL_FLAGS
        if flag not in {"--skip-body-relative-render", "--skip-body-relative"}
    ]
    monkeypatch.setattr(sys, "argv", ["verify_stack.py", *flags])

    assert verifier.main() == 0
    assert calls == [
        ("body-relative teleop probe", ["uv", "run", "python", "scripts/check_body_relative.py"]),
        (
            "body-relative Unity render payload probe",
            ["uv", "run", "python", "scripts/check_body_relative_render.py"],
        ),
    ]
