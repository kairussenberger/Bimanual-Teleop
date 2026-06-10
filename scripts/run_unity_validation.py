#!/usr/bin/env python
"""Run the TeleopRenderer Unity Editor validation when Unity is installed.

This is intentionally separate from `verify_stack.py`'s default hardware-free gate:
the current development machine may not have Unity Editor, but machines that do can
run this in batch mode to prove the C# DTOs compile and the primitive renderers can
apply a representative render state.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO / "unity" / "TeleopRenderer"
SUCCESS_MARKER = "TeleopRenderer editor validation passed"
DEFAULT_TIMEOUT_SECONDS = 180.0


def find_unity() -> str | None:
    env = os.environ.get("UNITY_EDITOR")
    if env and Path(env).exists():
        return env
    for name in ("Unity", "unity"):
        found = shutil.which(name)
        if found:
            return found
    app_root = Path("/Applications/Unity/Hub/Editor")
    candidates = sorted(app_root.glob("*/Unity.app/Contents/MacOS/Unity"), reverse=True)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    direct = Path("/Applications/Unity/Unity.app/Contents/MacOS/Unity")
    if direct.exists():
        return str(direct)
    return None


def tail(path: Path, lines: int = 80) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return "<no Unity log was written>"
    return "\n".join(text.splitlines()[-lines:])


def log_contains(path: Path, marker: str) -> bool:
    try:
        return marker in path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unity", help="path to Unity executable; defaults to UNITY_EDITOR, PATH, then Unity Hub paths")
    ap.add_argument("--require", action="store_true", help="fail if Unity Editor is not installed")
    ap.add_argument("--log-file", type=Path, help="Unity batchmode log path")
    ap.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS,
                    help="fail if Unity batch validation runs longer than this many seconds")
    args = ap.parse_args()

    unity = args.unity or find_unity()
    if not unity:
        msg = "Unity Editor not found; skipping Unity batch validation"
        if args.require:
            print(msg, file=sys.stderr)
            return 2
        print(msg)
        return 0

    log_file = args.log_file
    temp_dir = None
    if log_file is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="teleop-unity-")
        log_file = Path(temp_dir.name) / "unity-validation.log"

    cmd = [
        unity,
        "-batchmode",
        "-nographics",
        "-quit",
        "-projectPath", str(PROJECT),
        "-executeMethod", "TeleopEditorValidation.Run",
        "-logFile", str(log_file),
    ]
    print(" ".join(cmd), flush=True)
    try:
        proc = subprocess.run(cmd, cwd=REPO, timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired:
        print(f"\nUnity validation timed out after {args.timeout_seconds:.1f}s; log tail:", file=sys.stderr)
        print(tail(log_file), file=sys.stderr)
        if temp_dir is not None:
            temp_dir.cleanup()
        return 4
    if proc.returncode != 0:
        print("\nUnity validation failed; log tail:", file=sys.stderr)
        print(tail(log_file), file=sys.stderr)
        if temp_dir is not None:
            temp_dir.cleanup()
        return proc.returncode

    if not log_contains(log_file, SUCCESS_MARKER):
        print("\nUnity validation did not report the editor validation success marker; log tail:", file=sys.stderr)
        print(tail(log_file), file=sys.stderr)
        if temp_dir is not None:
            temp_dir.cleanup()
        return 3

    print("Unity batch validation passed")
    print(tail(log_file, lines=20))
    if temp_dir is not None:
        temp_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
