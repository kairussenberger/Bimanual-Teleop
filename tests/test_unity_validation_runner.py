from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_runner():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_unity_validation.py"
    spec = importlib.util.spec_from_file_location("run_unity_validation", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_unity_validation_runner_skips_when_unity_is_optional(monkeypatch, capsys):
    runner = _load_runner()
    monkeypatch.setattr(runner, "find_unity", lambda: None)
    monkeypatch.setattr(sys, "argv", ["run_unity_validation.py"])

    assert runner.main() == 0
    assert "Unity Editor not found; skipping" in capsys.readouterr().out


def test_unity_validation_runner_requires_unity_when_requested(monkeypatch, capsys):
    runner = _load_runner()
    monkeypatch.setattr(runner, "find_unity", lambda: None)
    monkeypatch.setattr(sys, "argv", ["run_unity_validation.py", "--require"])

    assert runner.main() == 2
    assert "Unity Editor not found; skipping" in capsys.readouterr().err


def test_unity_validation_runner_rejects_missing_success_marker(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    log_file = tmp_path / "unity.log"

    def fake_run(cmd, cwd, timeout):
        assert timeout == runner.DEFAULT_TIMEOUT_SECONDS
        log_file.write_text("Unity exited cleanly without running validation\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_unity_validation.py", "--unity", "/tmp/fake-unity", "--log-file", str(log_file)],
    )

    assert runner.main() == 3
    assert "success marker" in capsys.readouterr().err


def test_unity_validation_runner_accepts_success_marker(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    log_file = tmp_path / "unity.log"

    def fake_run(cmd, cwd, timeout):
        assert timeout == 12.5
        log_file.write_text(f"{runner.SUCCESS_MARKER}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_unity_validation.py", "--unity", "/tmp/fake-unity", "--log-file", str(log_file), "--timeout-seconds", "12.5"],
    )

    assert runner.main() == 0
    assert "Unity batch validation passed" in capsys.readouterr().out


def test_unity_validation_runner_times_out_cleanly(tmp_path, monkeypatch, capsys):
    runner = _load_runner()
    log_file = tmp_path / "unity.log"

    def fake_run(cmd, cwd, timeout):
        log_file.write_text("Unity is still importing assets\n", encoding="utf-8")
        raise runner.subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_unity_validation.py", "--unity", "/tmp/fake-unity", "--log-file", str(log_file), "--timeout-seconds", "1.25"],
    )

    assert runner.main() == 4
    err = capsys.readouterr().err
    assert "timed out after 1.2s" in err
    assert "Unity is still importing assets" in err
