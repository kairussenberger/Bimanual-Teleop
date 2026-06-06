"""Smoke test for the synthetic harness (scripts/run_synthetic.py): it must import
and PASS a short headless verification — line + circle + pure roll — confirming the
two-stage IK tracks and that pure roll lands on j6. Keeps the DoD honest in CI.

    uv run pytest tests/test_synthetic.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _args(**over):
    base = dict(view=False, rerun=False, gif=None, no_gif=True, csv=None,
                hz=120.0, seconds=1.6, fps=15.0, width=320, height=240, traj=None)
    base.update(over)
    return SimpleNamespace(**base)


def test_synthetic_runs_and_all_pass_headless():
    import run_synthetic as rs
    rc = rs.run(_args(traj=["line", "circle", "roll", "pitch", "yaw"]))
    assert rc == 0, "a synthetic trajectory failed its tracking/limit/velocity check"


def test_synthetic_pure_roll_lands_on_j6():
    """Build the roll result directly and assert it is realised on j6 (index 5) —
    the central claim of the J6 isolation test."""
    import run_synthetic as rs
    rig = rs.load_rig()
    arm = rs.SyntheticArm(rig, "right")
    traj = next(t for t in rs.make_trajectories() if t.name == "roll")
    res = rs.TrajResult("right", "roll")
    q0 = arm.ik.q.copy()
    hz, n = 120.0, int(2.0 * 120)
    for k in range(n):
        t = k / hz
        arm.ik.solve(arm.target(traj, t))
        res.peak_dq = rs.np.maximum(res.peak_dq, rs.np.abs(arm.ik.q - q0))
    assert res.roll_joint == 5            # j6 carries the roll
    assert res.peak_dq[5] > res.peak_dq[3] and res.peak_dq[5] > res.peak_dq[4]
