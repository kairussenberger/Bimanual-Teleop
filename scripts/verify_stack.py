#!/usr/bin/env python
"""Hardware-free acceptance gate for the reworked teleop stack.

Runs the checks that prove the current repo can:
  1. pass the Python unit/integration suite,
  2. preserve the default rig contract for body-relative Unity teleop,
  3. keep runtime imports and dependencies MuJoCo-free,
  4. prove torso-relative wrist vectors drive arm motion,
  5. prove the Unity render payload carries the same body-relative command motion,
  6. keep the Pinocchio YAM model tied to the measured MJCF source geometry,
  7. track synthetic YAM arm trajectories without MuJoCo,
  8. keep the Unity C# DTO/render contract and sample fixture aligned with Python's payload,
  9. parse the render/hardware/Quest diagnostic CLIs,
  10. start the headless teleop loop,
  11. record and replay a deterministic session,
  12. publish the Unity TCP JSON stream that the Unity scaffold consumes.
Optionally, with `--unity-editor`, it also runs the Unity project in batch mode
and executes the Editor-side renderer validation.

This intentionally does not require a Quest, Unity Editor, SocketCAN, or hardware.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def run_step(name: str, cmd: list[str]) -> None:
    print(f"\n== {name} ==")
    print(" ".join(cmd), flush=True)
    t0 = time.monotonic()
    subprocess.run(cmd, cwd=REPO, check=True)
    print(f"ok: {name} ({time.monotonic() - t0:.1f}s)", flush=True)


def run_json_monitor_smoke(seconds: float) -> None:
    name = "Unity JSON monitor smoke"
    print(f"\n== {name} ==")
    teleop_cmd = [
        "uv", "run", "python", "-m", "bimanual_teleop.launch.run_teleop",
        "--vr", "fake", "--calib-seconds", "0", "--duration", str(max(seconds + 1.5, 2.0)),
    ]
    monitor_cmd = [
        "uv", "run", "python", "scripts/render_monitor.py",
        "--transport", "json", "--seconds", str(seconds),
        "--require-hand-render", "--require-bimanual-state", "--require-command-target", "--require-frame",
    ]
    print(" ".join(teleop_cmd), flush=True)
    t0 = time.monotonic()
    proc = subprocess.Popen(teleop_cmd, cwd=REPO)
    try:
        time.sleep(0.8)
        print(" ".join(monitor_cmd), flush=True)
        subprocess.run(monitor_cmd, cwd=REPO, check=True)
        rc = proc.wait(timeout=5.0)
        if rc != 0:
            raise subprocess.CalledProcessError(rc, teleop_cmd)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    print(f"ok: {name} ({time.monotonic() - t0:.1f}s)", flush=True)


def run_record_replay_smoke(seconds: float) -> None:
    name = "record/replay launch smoke"
    print(f"\n== {name} ==")
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="bimanual-replay-") as td:
        session = str(Path(td) / "session.npz")
        record_cmd = [
            "uv", "run", "python", "-m", "bimanual_teleop.launch.run_teleop",
            "--vr", "fake", "--duration", str(seconds), "--record", session,
        ]
        replay_cmd = [
            "uv", "run", "python", "-m", "bimanual_teleop.launch.run_teleop",
            "--vr", "replay", session, "--duration", str(seconds),
        ]
        print(" ".join(record_cmd), flush=True)
        subprocess.run(record_cmd, cwd=REPO, check=True)
        if not Path(session).exists():
            raise FileNotFoundError(session)
        print(" ".join(replay_cmd), flush=True)
        subprocess.run(replay_cmd, cwd=REPO, check=True)
    print(f"ok: {name} ({time.monotonic() - t0:.1f}s)", flush=True)


def run_launch_cli_smoke() -> None:
    run_step("launch CLI help", [
        "uv", "run", "python", "-m", "bimanual_teleop.launch.run_teleop", "--help",
    ])
    run_step("hardware CLI help", [
        "uv", "run", "python", "-m", "bimanual_teleop.launch.run_hw", "--help",
    ])
    run_step("Quest diagnostic CLI help", [
        "uv", "run", "python", "scripts/check_quest.py", "--help",
    ])
    run_step("Quest roll diagnostic CLI help", [
        "uv", "run", "python", "scripts/check_roll.py", "--help",
    ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-pytest", action="store_true")
    ap.add_argument("--skip-rig-contract", action="store_true")
    ap.add_argument("--skip-no-mujoco-runtime", action="store_true")
    ap.add_argument("--skip-body-relative", action="store_true")
    ap.add_argument("--skip-body-relative-render", action="store_true")
    ap.add_argument("--skip-yam-geometry", action="store_true")
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--skip-unity-contract", action="store_true")
    ap.add_argument("--skip-cli", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--skip-record-replay", action="store_true")
    ap.add_argument("--skip-json-monitor", action="store_true")
    ap.add_argument("--unity-editor", action="store_true",
                    help="also run Unity Editor batch validation; requires Unity installed")
    ap.add_argument("--smoke-seconds", type=float, default=0.25)
    ap.add_argument("--monitor-seconds", type=float, default=0.25)
    args = ap.parse_args()

    try:
        if not args.skip_pytest:
            run_step("pytest", ["uv", "run", "pytest", "-q"])
        if not args.skip_rig_contract:
            run_step("rig contract", ["uv", "run", "python", "scripts/check_rig_contract.py"])
        if not args.skip_no_mujoco_runtime:
            run_step("no MuJoCo runtime", ["uv", "run", "python", "scripts/check_no_mujoco_runtime.py"])
        if not args.skip_body_relative:
            run_step("body-relative teleop probe", ["uv", "run", "python", "scripts/check_body_relative.py"])
        if not args.skip_body_relative_render:
            run_step("body-relative Unity render payload probe", [
                "uv", "run", "python", "scripts/check_body_relative_render.py",
            ])
        if not args.skip_yam_geometry:
            run_step("YAM geometry provenance", ["uv", "run", "python", "scripts/check_yam_geometry.py"])
        if not args.skip_synthetic:
            run_step("synthetic IK trajectories", ["uv", "run", "python", "scripts/run_synthetic.py"])
        if not args.skip_unity_contract:
            run_step("Unity render contract", ["uv", "run", "python", "scripts/check_unity_contract.py"])
        if not args.skip_cli:
            run_launch_cli_smoke()
        if not args.skip_smoke:
            run_step("headless teleop smoke", [
                "uv", "run", "python", "-m", "bimanual_teleop.launch.run_teleop",
                "--vr", "fake", "--duration", str(args.smoke_seconds),
            ])
        if not args.skip_record_replay:
            run_record_replay_smoke(args.smoke_seconds)
        if not args.skip_json_monitor:
            run_json_monitor_smoke(args.monitor_seconds)
        if args.unity_editor:
            run_step("Unity Editor batch validation", [
                "uv", "run", "python", "scripts/run_unity_validation.py", "--require",
            ])
    except subprocess.CalledProcessError as e:
        print(f"\nFAILED: {e.cmd} exited with {e.returncode}", file=sys.stderr)
        return e.returncode

    print("\nstack verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
