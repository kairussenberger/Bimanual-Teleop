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


def _is_rotation(R, atol=1e-9, det_tol=1e-7):
    R = np.asarray(R, float)
    return (np.allclose(R.T @ R, np.eye(3), atol=atol)
            and abs(np.linalg.det(R) - 1.0) < det_tol)


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
    from bimanual_teleop.vr.frames import SE3, SO3
    return SE3.from_rotation_and_translation(SO3.from_matrix(np.asarray(R, float)), np.asarray(p, float))


def test_clutch_orientation_world_frame_contract():
    """Orientation obeys the SAME change of basis as translation: a wrist rotation
    of θ about ctrl-frame axis `a` since engage produces an EE delta of θ about the
    base-frame axis `R·a`, applied about the anchor in the base frame — from ANY
    anchor orientations, with no stance/orientation calibration."""
    rng = np.random.default_rng(7)
    R = F.euler_to_R([0.2, -0.7, 1.4])              # arbitrary proper ctrl→base map
    for _ in range(5):
        anchor_ctrl_R = F.euler_to_R(rng.uniform(-2, 2, 3))
        anchor_ee_R = F.euler_to_R(rng.uniform(-2, 2, 3))
        a = rng.normal(size=3)
        a /= np.linalg.norm(a)
        th = rng.uniform(-2.0, 2.0)
        m = F.ClutchMapper(R, pos_scale=1.0)
        m.engage(_se3(anchor_ctrl_R, [1, 2, 3]), _se3(anchor_ee_R, [0.3, 0.1, 0.5]))
        ctrl_now = _se3(_axis_angle_R(a, th) @ anchor_ctrl_R, [1, 2, 3])   # left/world delta
        tgt = m.target(ctrl_now)
        D_base = tgt.rotation().as_matrix() @ anchor_ee_R.T
        assert _is_rotation(D_base)
        assert np.allclose(D_base, _axis_angle_R(R @ a, th), atol=1e-9)


def test_clutch_orientation_body_relative_real_rig_axes():
    """THE regression test for the scrambled-wrist bug (measured ≈145° median axis
    error on a real Quest session): through the REAL body-relative data path —
    head_op_axes (a left-handed basis), body_relative_hand_sample, R_base_from_body
    on each side's real base_quat — a physical hand rotation about a body axis must
    command an EE rotation about the corresponding ROBOT-WORLD axis, same angle,
    right-hand rule, for BOTH arms and ANY head yaw. The two reflections (operator
    basis and W_AXES) must cancel exactly."""
    from bimanual_teleop.config import load_rig
    from bimanual_teleop.vr.calibrate import R_base_from_body, body_relative_hand_sample, head_op_axes

    rig = load_rig()
    head = np.eye(4)
    head[:3, :3] = _axis_angle_R([0, 1, 0], 0.5)    # operator turned 0.5 rad in the room
    head[:3, 3] = [0.0, 1.6, 0.0]
    op = head_op_axes(head)                          # [right, up, forward] columns in WebXR
    # physical rotation axis in the room ↔ expected robot-world rotation axis
    cases = [(op[:, 2], np.array([-1.0, 0.0, 0.0])),   # body forward → world −X
             (op[:, 0], np.array([0.0, 1.0, 0.0])),    # body right   → world +Y
             (op[:, 1], np.array([0.0, 0.0, 1.0]))]    # body up      → world +Z
    W0 = np.eye(4)
    W0[:3, :3] = F.euler_to_R([0.9, -0.3, 0.4])      # arbitrary raw wrist orientation
    W0[:3, 3] = [0.2, 1.2, -0.4]
    th = 0.9
    for side in ("left", "right"):
        bq = rig["arms"][side]["base_quat"]
        base_R = F.quat_to_R(bq)
        m = F.ClutchMapper(R_base_from_body(bq), pos_scale=1.0)
        for axis_xr, want_world in cases:
            s0 = body_relative_hand_sample(F.HandSample(tracked=True, wrist=W0), head)
            anchor_ee_R = F.euler_to_R([0.3, -0.4, 0.8])
            m.engage(F.mat_to_se3(s0.wrist), _se3(anchor_ee_R, [0.3, 0.1, 0.5]))
            W1 = W0.copy()
            W1[:3, :3] = _axis_angle_R(axis_xr, th) @ W0[:3, :3]   # physical room rotation
            s1 = body_relative_hand_sample(F.HandSample(tracked=True, wrist=W1), head)
            tgt = m.target(F.mat_to_se3(s1.wrist))
            D_world = base_R @ (tgt.rotation().as_matrix() @ anchor_ee_R.T) @ base_R.T
            # rig.yaml base quats carry ~5 decimals; base_R appears 4x in this chain,
            # so the result is orthogonal/correct only to ~1e-4
            assert _is_rotation(D_world, atol=1e-4, det_tol=1e-3)
            assert np.allclose(D_world, _axis_angle_R(want_world, th), atol=1e-4), (side, want_world)


def test_clutch_orientation_body_turn_invariance():
    """Turning your whole body (head yaw, hand rigid relative to the torso) must not
    move OR rotate the EE target — the orientation analog of the translation
    body-relative invariance the probes already check."""
    from bimanual_teleop.config import load_rig
    from bimanual_teleop.vr.calibrate import R_base_from_body, body_relative_hand_sample

    rig = load_rig()
    bq = rig["arms"]["right"]["base_quat"]
    m = F.ClutchMapper(R_base_from_body(bq), pos_scale=1.0)
    head0 = np.eye(4)
    head0[:3, 3] = [0.0, 1.6, 0.0]
    W0 = np.eye(4)
    W0[:3, :3] = F.euler_to_R([0.4, 0.2, -0.7])
    W0[:3, 3] = [0.25, 1.3, -0.45]
    s0 = body_relative_hand_sample(F.HandSample(tracked=True, wrist=W0), head0)
    anchor_ee_R = F.euler_to_R([0.3, -0.4, 0.8])
    m.engage(F.mat_to_se3(s0.wrist), _se3(anchor_ee_R, [0.3, 0.1, 0.5]))

    Q = np.eye(4)
    Q[:3, :3] = _axis_angle_R([0, 1, 0], 0.8)        # body turn about the spine axis
    head1 = Q @ head0
    W1 = Q @ W0                                       # hand carried rigidly with the body
    s1 = body_relative_hand_sample(F.HandSample(tracked=True, wrist=W1), head1)
    tgt = m.target(F.mat_to_se3(s1.wrist))
    assert np.allclose(tgt.rotation().as_matrix(), anchor_ee_R, atol=1e-9)
    assert np.allclose(tgt.translation(), [0.3, 0.1, 0.5], atol=1e-9)


def test_clutch_position_change_of_basis():
    """Position delta is rotated into the base frame by R (not copied raw): a +x
    wrist move under R = WEBXR_TO_WORLD must push the EE +Y in base coords."""
    m = F.ClutchMapper(F.WEBXR_TO_WORLD, pos_scale=1.0)
    m.engage(_se3(np.eye(3), [0, 0, 0]), _se3(np.eye(3), [0.1, 0.2, 0.3]))
    tgt = m.target(_se3(np.eye(3), [0.05, 0, 0]))   # +5cm webxr x (right)
    assert np.allclose(tgt.translation(), [0.1, 0.25, 0.3], atol=1e-9)  # -> +Y base


def test_intrinsic_twist_pure_pronation_is_pure_ee_tool_roll():
    """THE wrist-singularity killer: a wrist turn about YOUR forearm axis commands
    a PURE rotation about the EE's OWN tool/j6 axis — same magnitude, zero swing —
    from ANY anchor orientation and ANY hand attitude, both sides."""
    from bimanual_teleop.arms.ik import ArmIK
    from bimanual_teleop.config import load_rig
    from bimanual_teleop.vr.calibrate import R_base_from_body, body_relative_hand_sample

    rig = load_rig()
    h_local = np.array(rig["mapping"]["hand_twist_axis"], dtype=float)
    h_local /= np.linalg.norm(h_local)
    head = np.eye(4)
    head[:3, 3] = [0.0, 1.6, 0.0]
    rng = np.random.default_rng(3)
    for side in ("left", "right"):
        ik = ArmIK(rig, side)
        m = F.ClutchMapper(R_base_from_body(rig["arms"][side]["base_quat"]),
                           twist_mode="intrinsic", hand_twist_axis=h_local,
                           ee_tool_axis=ik.ee_tool_axis_local)
        for _ in range(4):
            W0 = np.eye(4)
            W0[:3, :3] = F.euler_to_R(rng.uniform(-1.5, 1.5, 3))
            W0[:3, 3] = [0.2, 1.3, -0.4]
            A = F.euler_to_R(rng.uniform(-2, 2, 3))
            th = rng.uniform(-1.2, 1.2)
            a_phys = W0[:3, :3] @ h_local                      # forearm axis in the room
            W1 = W0.copy()
            W1[:3, :3] = _axis_angle_R(a_phys, th) @ W0[:3, :3]
            s0 = body_relative_hand_sample(F.HandSample(tracked=True, wrist=W0), head)
            s1 = body_relative_hand_sample(F.HandSample(tracked=True, wrist=W1), head)
            m.engage(F.mat_to_se3(s0.wrist), _se3(A, [0.3, 0.1, 0.5]), t=0.0)
            D = m.target(F.mat_to_se3(s1.wrist), 0.0).rotation().as_matrix() @ A.T
            rv = F.rotvec(D)
            ang = np.linalg.norm(rv)
            # rig base quats carry ~5 decimals → ~1e-5 rad numerical floor
            assert abs(ang - abs(th)) < 1e-4                   # full magnitude preserved
            axis = rv / ang
            tool_b = A @ ik.ee_tool_axis_local                 # EE tool axis in base
            assert abs(abs(axis @ tool_b) - 1.0) < 1e-6, (side, axis, tool_b)  # PURE tool roll


def test_intrinsic_twist_equals_world_mapping_when_axes_align():
    """Sign pin: when the EE tool axis happens to point along the world-mapped
    twist axis, intrinsic and world modes must produce the SAME target."""
    from bimanual_teleop.arms.ik import ArmIK
    from bimanual_teleop.config import load_rig
    from bimanual_teleop.vr.calibrate import R_base_from_body, body_relative_hand_sample

    def rot_a_to_b(a, b):
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        v = np.cross(a, b)
        c = float(a @ b)
        if np.linalg.norm(v) < 1e-9:
            return np.eye(3)
        return _axis_angle_R(v, np.arctan2(np.linalg.norm(v), c))

    rig = load_rig()
    h_local = np.array(rig["mapping"]["hand_twist_axis"], dtype=float)
    h_local /= np.linalg.norm(h_local)
    head = np.eye(4)
    head[:3, 3] = [0.0, 1.6, 0.0]
    for side in ("left", "right"):
        ik = ArmIK(rig, side)
        Rmap = R_base_from_body(rig["arms"][side]["base_quat"])
        W0 = np.eye(4)
        W0[:3, :3] = F.euler_to_R([0.5, -0.2, 0.3])
        W0[:3, 3] = [0.2, 1.3, -0.4]
        W1 = W0.copy()
        W1[:3, :3] = _axis_angle_R(W0[:3, :3] @ h_local, 0.7) @ W0[:3, :3]
        s0 = body_relative_hand_sample(F.HandSample(tracked=True, wrist=W0), head)
        s1 = body_relative_hand_sample(F.HandSample(tracked=True, wrist=W1), head)
        ctrl0, ctrl1 = F.mat_to_se3(s0.wrist), F.mat_to_se3(s1.wrist)
        dR = ctrl1.rotation().as_matrix() @ ctrl0.rotation().as_matrix().T
        rv = F.rotvec(Rmap @ dR @ Rmap.T)
        A = rot_a_to_b(ik.ee_tool_axis_local, rv / np.linalg.norm(rv))
        ee = _se3(A, [0.3, 0.1, 0.5])
        mw = F.ClutchMapper(Rmap, twist_mode="world")
        mi = F.ClutchMapper(Rmap, twist_mode="intrinsic", hand_twist_axis=h_local,
                            ee_tool_axis=ik.ee_tool_axis_local)
        mw.engage(ctrl0, ee, t=0.0)
        mi.engage(ctrl0, ee, t=0.0)
        Rw = mw.target(ctrl1, 0.0).rotation().as_matrix()
        Ri = mi.target(ctrl1, 0.0).rotation().as_matrix()
        assert np.degrees(np.linalg.norm(F.rotvec(Rw.T @ Ri))) < 1e-6, side


def test_intrinsic_twist_continuity_and_body_turn_invariance():
    from bimanual_teleop.arms.ik import ArmIK
    from bimanual_teleop.config import load_rig
    from bimanual_teleop.vr.calibrate import R_base_from_body, body_relative_hand_sample

    rig = load_rig()
    ik = ArmIK(rig, "right")
    m = F.ClutchMapper(R_base_from_body(rig["arms"]["right"]["base_quat"]),
                       twist_mode="intrinsic",
                       hand_twist_axis=rig["mapping"]["hand_twist_axis"],
                       ee_tool_axis=ik.ee_tool_axis_local)
    head0 = np.eye(4)
    head0[:3, 3] = [0.0, 1.6, 0.0]
    W0 = np.eye(4)
    W0[:3, :3] = F.euler_to_R([0.4, 0.2, -0.7])
    W0[:3, 3] = [0.25, 1.3, -0.45]
    s0 = body_relative_hand_sample(F.HandSample(tracked=True, wrist=W0), head0)
    A = F.euler_to_R([0.3, -0.4, 0.8])
    m.engage(F.mat_to_se3(s0.wrist), _se3(A, [0.3, 0.1, 0.5]), t=0.0)
    assert np.allclose(m.target(F.mat_to_se3(s0.wrist), 0.0).rotation().as_matrix(), A, atol=1e-9)
    Q = np.eye(4)
    Q[:3, :3] = _axis_angle_R([0, 1, 0], 0.8)
    s1 = body_relative_hand_sample(F.HandSample(tracked=True, wrist=Q @ W0), Q @ head0)
    assert np.allclose(m.target(F.mat_to_se3(s1.wrist), 1.0).rotation().as_matrix(), A, atol=1e-9)
