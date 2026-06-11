"""Motion guardrails: target governor (speed cap, teleport rejection, angular
rate cap), the in-loop joint command shaper, and the hand-capsule separation.

Sized from a real session (recordings/live_0611_130501.npz): operator motion
peaks ≈2 m/s, tracking glitches 58–65 m/s, and the left-wrist-roll singularity
drove j4/j6 at 20–25 rad/s before the shaper entered the sim path.

    uv run pytest tests/test_guardrails.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from bimanual_teleop.arms.arm_control import ArmController
from bimanual_teleop.config import SIDES, load_rig
from bimanual_teleop.safety.separation import closest_points_segments, separate_capsules
from bimanual_teleop.vr.frames import HandSample

DT = 1.0 / 120.0


def _hs(p, R=None) -> HandSample:
    W = np.eye(4)
    if R is not None:
        W[:3, :3] = R
    W[:3, 3] = np.asarray(p, dtype=float)
    return HandSample(tracked=True, wrist=W)


def _rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# --------------------------------------------------------------------------- #
# target governor
# --------------------------------------------------------------------------- #
def test_governor_caps_target_speed():
    rig = load_rig()
    ac = ArmController(rig, "right")
    v_max = float(rig["safety"]["target_speed_max"])
    t, p = 0.0, np.array([0.25, 0.0, 0.30])
    prev = None
    for i in range(240):                       # 2 s sweep at ~1.8 m/s requested
        p = p + np.array([0.0, 0.015, 0.0])    # 1.8 m/s in body axes
        plan = ac.plan(_hs(p), True, t)
        assert plan is not None
        if prev is not None:
            step = float(np.linalg.norm(plan["pw"] - prev))
            assert step <= v_max * DT * 1.05 + 1e-9, f"target moved {step/DT:.2f} m/s"
        prev = plan["pw"]
        t += DT

def test_governor_rejects_teleport_and_recovers():
    rig = load_rig()
    ac = ArmController(rig, "right")
    t = 0.0
    for _ in range(60):                        # settle engaged
        ac.plan(_hs([0.25, 0.0, 0.30]), True, t)
        t += DT
    assert ac.mapper.engaged
    # 0.6 m in one tick = 72 m/s — a tracking glitch, not a movement
    plan = ac.plan(_hs([0.25, 0.6, 0.30]), True, t)
    assert plan is None                        # the movement does not happen
    assert not ac.mapper.engaged               # re-anchored → glide on next tick
    t += DT
    plan = ac.plan(_hs([0.25, 0.6, 0.30]), True, t)
    assert plan is not None                    # following resumes (snap-free)


def test_governor_caps_angular_speed():
    rig = load_rig()
    ac = ArmController(rig, "right")
    w_max = float(rig["safety"]["target_ang_speed_max"])
    t = 0.0
    for _ in range(60):
        ac.plan(_hs([0.25, 0.0, 0.30]), True, t)
        t += DT
    prev = None
    for i in range(120):                       # demand a fast 180°/s… × 4 roll
        R = _rotx(12.0 * (i + 1) * DT)         # 12 rad/s requested
        plan = ac.plan(_hs([0.25, 0.0, 0.30], R), True, t)
        R_w = ac.base_R @ plan["R"].as_matrix()
        if prev is not None:
            d = prev.T @ R_w
            ang = float(np.arccos(np.clip((np.trace(d) - 1) / 2, -1, 1)))
            assert ang <= w_max * DT * 1.05 + 1e-6, f"attitude turned {ang/DT:.1f} rad/s"
        prev = R_w
        t += DT


# --------------------------------------------------------------------------- #
# in-loop joint shaper
# --------------------------------------------------------------------------- #
def test_shaper_bounds_joint_speed_through_violent_input():
    rig = load_rig()
    ac = ArmController(rig, "right")
    rate = float(rig["safety"]["sim_rate_limit"])
    t, q_prev = 0.0, None
    rng = np.random.default_rng(0)
    p = np.array([0.25, 0.0, 0.30])
    for i in range(480):                       # 4 s of erratic, fast, rolling input
        if i % 40 == 0:
            p = np.array([0.25, 0, 0.30]) + rng.uniform(-0.25, 0.25, 3)
        R = _rotx(rng.uniform(-2.5, 2.5))
        q = ac.update(_hs(p, R), True, t)
        if q_prev is not None:
            dq = np.abs(q - q_prev) / DT
            assert dq.max() <= rate + 1e-6, f"joint moved {dq.max():.1f} rad/s at tick {i}"
        q_prev = q
        t += DT


def test_shaper_keeps_render_state_consistent():
    """The shaped pose is seeded back: FK (render stream) and the returned
    command must be the same robot."""
    rig = load_rig()
    ac = ArmController(rig, "right")
    t = 0.0
    for _ in range(30):
        q = ac.update(_hs([0.3, 0.1, 0.35]), True, t)
        np.testing.assert_allclose(q, ac.ik.q, atol=1e-12)
        t += DT


# --------------------------------------------------------------------------- #
# capsule separation
# --------------------------------------------------------------------------- #
def test_capsules_fingers_pointing_at_each_other():
    """The case the point-pair guard missed: fingertips aimed at the other palm."""
    w_l = np.array([0.0, -0.15, 1.0])
    w_r = np.array([0.0, 0.15, 1.0])
    d_l = np.array([0.0, 1.0, 0.0])            # left fingers point right…
    d_r = np.array([0.0, -1.0, 0.0])           # …right fingers point left
    n_l, n_r = separate_capsules(w_l, w_r, d_l, d_r, 0.19, 0.12)
    cl, cr = closest_points_segments(n_l, n_l + d_l * 0.19, n_r, n_r + d_r * 0.19)
    assert np.linalg.norm(cr - cl) == pytest.approx(0.12, abs=1e-9)


def test_capsules_parallel_palms_and_noop():
    w_l = np.array([0.0, -0.04, 1.0])
    w_r = np.array([0.0, 0.04, 1.0])
    fwd = np.array([-1.0, 0.0, 0.0])
    n_l, n_r = separate_capsules(w_l, w_r, fwd, fwd, 0.19, 0.12)
    assert np.linalg.norm(n_r - n_l) == pytest.approx(0.12, abs=1e-9)
    far_l, far_r = np.array([0.0, -0.4, 1.0]), np.array([0.0, 0.4, 1.0])
    n_l, n_r = separate_capsules(far_l, far_r, fwd, fwd, 0.19, 0.12)
    np.testing.assert_allclose(n_l, far_l)
    np.testing.assert_allclose(n_r, far_r)


def test_closest_points_crossing_segments():
    cl, cr = closest_points_segments([-1, 0, 0], [1, 0, 0], [0, -1, 0.2], [0, 1, 0.2])
    np.testing.assert_allclose(cl, [0, 0, 0], atol=1e-9)
    np.testing.assert_allclose(cr, [0, 0, 0.2], atol=1e-9)


# --------------------------------------------------------------------------- #
# j6 saturation hysteresis (the left-roll pivot from live_0611_133349)
# --------------------------------------------------------------------------- #
def _roll_beyond_stop(side: str, direction: float):
    """Drive a pure tool-axis roll well past the j6 stop (beyond the ±π wrap of
    the relative twist) and back. Returns the joint trajectory."""
    from bimanual_teleop.arms.ik import ArmIK
    from bimanual_teleop.vr.frames import SE3, SO3
    rig = load_rig()
    ik = ArmIK(rig, side)
    R0 = ik.fk_ee().rotation().as_matrix()
    p0 = ik.fk_wrist().translation()
    a = ik._joint_axis_base(ik.joints[5])
    a = a / np.linalg.norm(a)

    def rot(th):
        c, s = np.cos(th), np.sin(th)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + s * K + (1 - c) * (K @ K)

    qs = []
    thetas = list(np.linspace(0, direction * 6.0, 120)) + \
             list(np.linspace(direction * 6.0, 0, 120))
    for th in thetas:
        ik.solve(SE3.from_rotation_and_translation(SO3.from_matrix(rot(th) @ R0), p0))
        qs.append(ik.q)
    return np.array(qs), np.asarray(rig["arms"][side]["neutral_q"], float)


def test_left_roll_past_wrap_stays_pinned_no_pivot():
    """The measured failure: left j6 has only ~30° of headroom rolling negative;
    past the ±π wrap the old code walked j6 the LONG way across its range
    through the wrist singularity while j4/j5 wandered ~0.5 rad."""
    qs, q_rest = _roll_beyond_stop("left", -1.0)
    settle = qs[20:]                       # after reaching the stop
    assert settle[:, 5].max() <= -1.55, \
        f"left j6 crossed back to {settle[:,5].max():+.2f} — long-way travel"
    for j in (3, 4):
        wander = np.abs(qs[:, j] - q_rest[j]).max()
        assert wander < 0.30, f"left j{j+1} pivoted {wander:.2f} rad during saturated roll"


def test_right_roll_past_wrap_stays_pinned_no_pivot():
    qs, q_rest = _roll_beyond_stop("right", +1.0)
    settle = qs[20:]
    assert settle[:, 5].min() >= +1.55, \
        f"right j6 crossed back to {settle[:,5].min():+.2f} — long-way travel"
    for j in (3, 4):
        wander = np.abs(qs[:, j] - q_rest[j]).max()
        assert wander < 0.30, f"right j{j+1} pivoted {wander:.2f} rad during saturated roll"
