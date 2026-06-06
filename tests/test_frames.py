"""Frame-transform unit tests — written FIRST, per CLAUDE.md Section 9 ("frame
transforms get a unit test before use") and the build spec's Section 3.

The headline test is the **+90° roll change-of-basis** (Section 3): a pure roll
about the operator's hand forward axis must come out as a pure roll about the
robot's tool axis — *and nothing else*. A wrong basis is exactly what scrambles a
clean wrist roll into a pitch+yaw mush, so this is the test that pins R_align.

    uv run pytest tests/test_frames.py -q
"""
from __future__ import annotations

import numpy as np

from bimanual_teleop.vr import frames as F


# --- small helpers used only by the tests ---------------------------------- #
def _axis_angle_R(axis, angle):
    """Reference Rodrigues rotation (independent of the impl under test)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _is_rotation(R, atol=1e-9):
    R = np.asarray(R, float)
    return (np.allclose(R.T @ R, np.eye(3), atol=atol)
            and abs(np.linalg.det(R) - 1.0) < 1e-7)


# --- quat_to_R / euler_to_R basics ----------------------------------------- #
def test_quat_to_R_identity_and_known():
    assert np.allclose(F.quat_to_R([1, 0, 0, 0]), np.eye(3))
    # +90° about world +Z (w,x,y,z): x->y, y->-x
    R = F.quat_to_R([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-9)
    assert np.allclose(R @ [0, 1, 0], [-1, 0, 0], atol=1e-9)
    assert _is_rotation(R)


def test_euler_to_R_single_axis_and_orthonormal():
    # XYZ euler with only X set == rotation about x
    assert np.allclose(F.euler_to_R([0.7, 0, 0]), _axis_angle_R([1, 0, 0], 0.7), atol=1e-9)
    assert np.allclose(F.euler_to_R([0, 0.4, 0]), _axis_angle_R([0, 1, 0], 0.4), atol=1e-9)
    assert _is_rotation(F.euler_to_R([0.3, -0.5, 1.1]))


def test_rotvec_roundtrips_known_rotations():
    for axis, ang in (([1, 0, 0], 0.9), ([0, 1, 0], -1.2), ([1, 1, 0], 0.5)):
        R = _axis_angle_R(axis, ang)
        v = F.rotvec(R)
        a = np.asarray(axis, float)
        a = a / np.linalg.norm(a)
        assert np.allclose(v, a * ang, atol=1e-7)
    assert np.allclose(F.rotvec(np.eye(3)), np.zeros(3))


# --- R_align = WEBXR_TO_WORLD (the static OpenXR->robot basis change) ------- #
def test_webxr_to_world_axis_mapping():
    """Spec: webxr +x -> world +Y, +y -> +Z, +z -> +X (robot faces world -X, so
    webxr forward -z -> world -X)."""
    R = F.WEBXR_TO_WORLD
    assert _is_rotation(R)
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])   # right  -> +Y
    assert np.allclose(R @ [0, 1, 0], [0, 0, 1])   # up     -> +Z
    assert np.allclose(R @ [0, 0, 1], [1, 0, 0])   # back   -> +X
    assert np.allclose(R @ [0, 0, -1], [-1, 0, 0])  # forward -> -X (robot forward)


def test_r_align_alias_is_webxr_to_world():
    # R_ALIGN is the named, documented handle for the static basis change.
    assert np.allclose(F.R_ALIGN, F.WEBXR_TO_WORLD)


def test_r_base_from_vr_identity_base_is_webxr_to_world():
    R = F.r_base_from_vr([1, 0, 0, 0])  # identity base quat
    assert np.allclose(R, F.WEBXR_TO_WORLD)
    assert _is_rotation(R)


# --- the change-of-basis helper (rotation + quaternion forms) --------------- #
def test_conjugate_rotation_is_change_of_basis():
    """conjugate_rotation(B, dR) = B dR Bᵀ re-expresses the rotation dR (given in
    one frame) in the frame B maps INTO. The axis transforms by B; the angle is
    preserved."""
    B = F.WEBXR_TO_WORLD
    dR = _axis_angle_R([0, 0, 1], 0.6)
    out = F.conjugate_rotation(B, dR)
    assert _is_rotation(out)
    # angle preserved
    assert np.allclose(np.linalg.norm(F.rotvec(out)), 0.6, atol=1e-9)
    # axis rotated by B
    assert np.allclose(F.rotvec(out), B @ F.rotvec(dR), atol=1e-9)


def test_change_of_basis_PLUS_90_ROLL_about_forward():
    """THE test (spec Section 3): a +90° roll about the operator's hand FORWARD
    axis (webxr -z) must become a +90° roll about the robot's TOOL/forward axis
    (world -X) — and carry NO pitch/yaw. A wrong R_align is what turns a clean
    roll into a pitch+yaw scramble; this assertion catches that."""
    theta = np.pi / 2
    hand_forward = np.array([0.0, 0.0, -1.0])          # webxr forward
    dR_head = _axis_angle_R(hand_forward, theta)        # pure +90° forearm roll

    dR_robot = F.conjugate_rotation(F.R_ALIGN, dR_head)

    v = F.rotvec(dR_robot)
    robot_forward = F.R_ALIGN @ hand_forward            # = world -X
    assert np.allclose(robot_forward, [-1, 0, 0], atol=1e-9)
    # magnitude is exactly the roll angle
    assert np.allclose(np.linalg.norm(v), theta, atol=1e-9)
    # the rotation axis is the robot tool axis — and NOTHING else
    assert np.allclose(v / np.linalg.norm(v), robot_forward, atol=1e-9)
    # explicit "nothing else": components orthogonal to forward (pitch/yaw) are ~0
    perp = v - (v @ robot_forward) * robot_forward
    assert np.linalg.norm(perp) < 1e-9
    # strongest form: the WHOLE matrix equals a hand-built +90° roll about world -X,
    # so the test fails if the conjugation is dropped/wrong (not just the axis).
    assert np.allclose(dR_robot, _axis_angle_R([-1, 0, 0], theta), atol=1e-9)


def test_quaternion_change_of_basis_matches_matrix():
    """The quaternion conjugation q_align · dq · q_align⁻¹ (spec Section 3) must
    agree with the matrix form B dR Bᵀ."""
    q_align = F.R_to_quat(F.R_ALIGN)
    dq = F.quat_from_axis_angle([0, 0, -1], np.pi / 2)
    dq_robot = F.change_basis_quat(q_align, dq)
    assert np.allclose(F.quat_to_R(dq_robot),
                       F.conjugate_rotation(F.R_ALIGN, F.quat_to_R(dq)), atol=1e-9)
    # ground truth (not impl-vs-impl): the conjugated axis is R_ALIGN @ (0,0,-1),
    # the angle is preserved, so the result is that exact axis-angle rotation.
    gt = _axis_angle_R(F.R_ALIGN @ [0, 0, -1], np.pi / 2)
    assert np.allclose(F.quat_to_R(dq_robot), gt, atol=1e-9)


# --- quaternion helper correctness ----------------------------------------- #
def test_quat_roundtrip_and_algebra():
    for axis, ang in (([1, 0, 0], 0.9), ([0, 1, 1], -1.3), ([1, 2, 3], 2.0)):
        R = _axis_angle_R(axis, ang)
        q = F.R_to_quat(R)
        assert np.allclose(F.quat_to_R(q), R, atol=1e-9)        # R -> q -> R
    a = F.quat_from_axis_angle([0, 0, 1], 0.5)
    b = F.quat_from_axis_angle([0, 0, 1], 0.7)
    # composing rotations about the same axis adds angles
    assert np.allclose(F.quat_to_R(F.quat_mul(a, b)),
                       _axis_angle_R([0, 0, 1], 1.2), atol=1e-9)
    # q · q⁻¹ = identity
    assert np.allclose(F.quat_mul(a, F.quat_inv(a)), [1, 0, 0, 0], atol=1e-9)


# --- ClutchMapper: the relative/clutch mapping respects the +90° roll -------- #
def _se3(R, p):
    import mink
    return mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(np.asarray(R, float)), np.asarray(p, float))


def test_clutch_pure_roll_maps_to_pure_ee_roll_P_identity():
    """End-to-end through ClutchMapper with the calibration correspondence P = I:
    a pure wrist roll since engage produces an EE target whose delta-from-anchor is
    the SAME pure roll about the matching EE-local axis — nothing else. This is the
    spec's '+90° roll in -> +90° tool roll out' assertion at the mapping layer."""
    anchor_ee_R = F.euler_to_R([0.3, -0.4, 0.8])   # arbitrary engaged EE orientation
    anchor_ctrl_R = F.euler_to_R([1.1, 0.2, -0.5])  # arbitrary engaged wrist orientation
    m = F.ClutchMapper(F.euler_to_R([0.2, 0.0, 1.4]), pos_scale=1.0, abs_orientation=True)
    m.engage(_se3(anchor_ctrl_R, [1, 2, 3]), _se3(anchor_ee_R, [0.3, 0.1, 0.5]))

    # roll the wrist +90° about its LOCAL x axis (ClutchMapper measures dR in the
    # hand-local frame): ctrl.R = anchor · Rx(90)
    local_axis = np.array([1.0, 0.0, 0.0])
    dR_local = _axis_angle_R(local_axis, np.pi / 2)
    ctrl_now = _se3(anchor_ctrl_R @ dR_local, [1, 2, 3])

    tgt = m.target(ctrl_now)
    # EE delta expressed in the EE-local frame
    ee_delta_local = anchor_ee_R.T @ tgt.rotation().as_matrix()
    v = F.rotvec(ee_delta_local)
    assert np.allclose(np.linalg.norm(v), np.pi / 2, atol=1e-7)        # 90° magnitude
    assert np.allclose(v / np.linalg.norm(v), local_axis, atol=1e-7)   # same axis, nothing else


def test_clutch_P_remaps_roll_axis():
    """The calibrated correspondence P (hand-local -> EE-local) is what steers WHICH
    EE axis a wrist roll drives. With P a 90° axis swap, a roll about hand-local x
    must come out as a roll about the remapped EE-local axis."""
    anchor_ee_R = np.eye(3)
    anchor_ctrl_R = np.eye(3)
    P = _axis_angle_R([0, 0, 1], np.pi / 2)        # swaps x<->y in EE-local
    m = F.ClutchMapper(np.eye(3))
    m.set_P(P)
    m.engage(_se3(anchor_ctrl_R, [0, 0, 0]), _se3(anchor_ee_R, [0, 0, 0]))
    dR_local = _axis_angle_R([1, 0, 0], np.pi / 2)  # roll about hand-local x
    tgt = m.target(_se3(anchor_ctrl_R @ dR_local, [0, 0, 0]))
    v = F.rotvec(tgt.rotation().as_matrix())
    assert np.allclose(v / np.linalg.norm(v), P @ [1, 0, 0], atol=1e-6)


def test_clutch_position_change_of_basis():
    """Position delta is rotated into the base frame by R (not copied raw): a +x
    wrist move under R = WEBXR_TO_WORLD must push the EE +Y in base coords."""
    m = F.ClutchMapper(F.WEBXR_TO_WORLD, pos_scale=1.0)
    m.engage(_se3(np.eye(3), [0, 0, 0]), _se3(np.eye(3), [0.1, 0.2, 0.3]))
    tgt = m.target(_se3(np.eye(3), [0.05, 0, 0]))   # +5cm webxr x (right)
    assert np.allclose(tgt.translation(), [0.1, 0.25, 0.3], atol=1e-9)  # -> +Y base
