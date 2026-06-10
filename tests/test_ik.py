"""pink two-stage IK wrapper tests: soft-limit enforcement (incl. the elbow), the
self-collision hook, and the CENTRAL J6 claim — a pure tool-axis roll is realised
on j6 (the wrist-roll joint), not by arcing the forearm. If the J6 test passes,
a clean wrist roll is an IK-solvable single-joint move, so any *real-wrist* roll
failure is a frames/tracking problem, not IK/limits (spec Section 7).

    uv run pytest tests/test_ik.py -q
"""
from __future__ import annotations

import pink
import numpy as np

from bimanual_teleop.arms.ik import ArmIK
from bimanual_teleop.config import load_rig
from bimanual_teleop.vr.frames import SE3, SO3, rotvec


def _axis_angle_R(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _ori_err_deg(R_ach, R_tgt):
    return float(np.degrees(np.linalg.norm(rotvec(R_ach.T @ R_tgt))))


def test_limit_margins_and_within_limits():
    ik = ArmIK(load_rig(), "left")
    m = ik.limit_margins(ik.q0)
    assert m.shape == (6,)
    assert ik.within_limits(ik.q0)                 # home sits inside the soft band
    assert not ik.within_limits(ik.soft_hi + 0.5)  # past the top soft limit
    assert not ik.within_limits(ik.soft_lo - 0.5)  # past the bottom soft limit


def test_collision_hook_off_by_default():
    """Default build = [ConfigurationLimit, VelocityLimit] per stage; the collision
    limit is opt-in (the standalone arm model has no collidable geoms)."""
    ik = ArmIK(load_rig(), "left")
    assert len(ik.limits_pos) == 2 and len(ik.limits_ori) == 2
    assert isinstance(ik.limits_pos[0], pink.limits.ConfigurationLimit)
    assert isinstance(ik.limits_pos[1], pink.limits.VelocityLimit)


def test_ik_never_hyperextends_elbow_under_sweep():
    """Drive the wrist in a circle around the workspace; the elbow (j3) and every
    joint stay within the human-plausible soft limits (invariant #5)."""
    ik = ArmIK(load_rig(), "left")
    ik.reset()
    p0 = ik.fk_wrist().translation()
    ee_R = ik.fk_ee().rotation()
    for k in range(72):
        ang = 2 * np.pi * k / 72
        dp = 0.12 * np.array([np.cos(ang), 0.0, np.sin(ang)])
        tgt = SE3.from_rotation_and_translation(ee_R, p0 + dp)
        ik.solve(tgt)
        assert ik.within_limits(), f"limit violation at step {k}: margins={ik.limit_margins().round(3)}"
        # explicit elbow check: j3 within its human-plausible soft range
        assert ik.soft_lo[ik.ELBOW] - 1e-6 <= ik.q[ik.ELBOW] <= ik.soft_hi[ik.ELBOW] + 1e-6


def test_pure_roll_is_realised_on_j6():
    """THE J6 isolation test. Roll the EE +theta about its OWN tool axis, holding
    wrist position. The wrist-roll joint j6 must carry essentially all of it, while
    j4/j5 (and the arm j1-j3) barely move — that's 'pronation/supination -> J6'."""
    ik = ArmIK(load_rig(), "left")
    ik.reset()
    q0 = ik.q.copy()
    wrist_p = ik.fk_wrist().translation()
    ee_R0 = ik.fk_ee().rotation().as_matrix()
    # Build the roll about the EE SITE's OWN axis (from FK), not the j6 joint axis,
    # so the test isn't circular — then independently assert that EE tool axis IS the
    # j6 axis (the physical reason a tool-axis roll lands on j6).
    j6_axis = ik._joint_axis_base(ik.joints[5])
    j6_axis = j6_axis / np.linalg.norm(j6_axis)
    align = np.abs(ee_R0.T @ j6_axis)          # which EE-local axis coincides with j6
    tool_col = int(np.argmax(align))
    assert align[tool_col] > 0.99, "EE site must have a local axis equal to the j6/tool axis"
    ee_tool = ee_R0[:, tool_col]               # the tool/roll axis, from the EE frame

    theta = 0.6                                # ~34°, inside the j6 soft cap
    tgt_R = _axis_angle_R(ee_tool, theta) @ ee_R0
    target = SE3.from_rotation_and_translation(SO3.from_matrix(tgt_R), wrist_p)
    for _ in range(200):
        ik.solve(target)

    dq = ik.q - q0
    assert _ori_err_deg(ik.fk_ee().rotation().as_matrix(), tgt_R) < 1.0   # orientation reached
    assert abs(dq[5]) > 0.97 * theta           # j6 carries (essentially all of) the roll
    assert abs(dq[3]) < 0.02 and abs(dq[4]) < 0.02   # wrist pitch/yaw barely move
    assert np.all(np.abs(dq[:3]) < 0.02)       # the ARM (j1-j3) does not arc to roll
    assert ik.within_limits()


def test_roll_beyond_j6_range_saturates_without_contortion():
    """When the commanded roll exceeds j6's travel, j6 must pin at its limit and
    the unrealizable remainder must be DROPPED — not smeared onto j4/j5 (the
    wrist-singularity contortion the swing–twist decomposition exists to kill)."""
    ik = ArmIK(load_rig(), "left")
    ik.reset()
    q0 = ik.q.copy()
    wrist_p = ik.fk_wrist().translation()
    ee_R0 = ik.fk_ee().rotation().as_matrix()
    j6_axis = ik._joint_axis_base(ik.joints[5])
    j6_axis = j6_axis / np.linalg.norm(j6_axis)
    align = np.abs(ee_R0.T @ j6_axis)
    ee_tool = ee_R0[:, int(np.argmax(align))]

    # Left j6 home is -90°; its NEGATIVE travel to the limit is only ~0.52 rad.
    headroom = float(q0[5] - ik.soft_lo[5])
    theta = -(headroom + 0.7)                       # demand ~40° more than exists
    sign = np.sign(ee_tool @ j6_axis) or 1.0        # tool axis vs joint axis sense
    tgt_R = _axis_angle_R(ee_tool, sign * theta) @ ee_R0
    target = SE3.from_rotation_and_translation(SO3.from_matrix(tgt_R), wrist_p)
    for _ in range(200):
        ik.solve(target)

    dq = ik.q - q0
    assert abs(dq[5] - (-headroom)) < 0.02          # j6 pinned at its limit
    assert abs(dq[3]) < 0.05 and abs(dq[4]) < 0.05  # NO swing contortion for roll
    assert np.all(np.abs(dq[:3]) < 0.02)            # arm does not arc
    assert ik.within_limits()
    # The residual orientation error equals the dropped twist (graceful saturation).
    err = _ori_err_deg(ik.fk_ee().rotation().as_matrix(), tgt_R)
    assert abs(err - np.degrees(0.7)) < 3.0


def test_twist_plus_swing_decomposition_converges():
    """A combined command (re-aim the tool axis AND roll about it) must converge,
    with the roll component carried by j6."""
    ik = ArmIK(load_rig(), "left")
    ik.reset()
    q0 = ik.q.copy()
    wrist_p = ik.fk_wrist().translation()
    ee_R0 = ik.fk_ee().rotation().as_matrix()
    j6_axis = ik._joint_axis_base(ik.joints[5])
    align = np.abs(ee_R0.T @ (j6_axis / np.linalg.norm(j6_axis)))
    tool_col = int(np.argmax(align))
    ee_tool = ee_R0[:, tool_col]
    swing_axis = ee_R0[:, (tool_col + 1) % 3]       # any EE axis ⊥ the tool axis

    twist, swing = 0.5, 0.3                          # both inside limits
    tgt_R = _axis_angle_R(swing_axis, swing) @ _axis_angle_R(ee_tool, twist) @ ee_R0
    target = SE3.from_rotation_and_translation(SO3.from_matrix(tgt_R), wrist_p)
    for _ in range(300):
        ik.solve(target)

    assert _ori_err_deg(ik.fk_ee().rotation().as_matrix(), tgt_R) < 2.0
    assert abs(abs(ik.q[5] - q0[5]) - twist) < 0.2   # j6 carries ~the twist
    assert ik.within_limits()
