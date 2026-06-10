"""Standalone single-arm **Pinocchio** model for the 6-DoF YAM, built programmatically
from the CAD-measured MJCF body tree (sim/models/yam_real/mjcf/yam_*_body.xml) so the
kinematics preserve the measured source geometry WITHOUT depending on MuJoCo at
runtime. This replaces the MuJoCo model that mink's ArmIK used.

Why programmatic (not the sibling URDF): the MJCF carries the CAD/ICP-calibrated j6
flange-normal axis + the real ±120° wrist range; a URDF revision on disk could drift.
The numbers below are transcribed from the MJCF source files and exercised by the
IK/render tests that use this Pinocchio model as the runtime kinematic authority.

Both physical arms are the SAME YAM unit (yam_right_body.xml header: "IDENTICAL
geometry, NOT an x-mirror"), so the joint chain is shared; only the EE site placement
(the ORCA flange) differs per side. The per-arm BASE pose is applied OUTSIDE the IK
(arm_control via base_R/base_pos), exactly as before — this model is in the arm's
own base frame.

Frame/placement bookkeeping (matching MuJoCo semantics):
  - A MuJoCo body has frame B_k = parent_body ∘ SE3(quat_k, pos_k); its hinge joint
    rotates about a line through `joint_pos_k` (in B_k) along `axis_k`.
  - Pinocchio's joint frame J_k rotates about ITS OWN origin, so we place
    J_k = B_k ∘ T(joint_pos_k):  jointPlacement_k = T(-parent_joint_pos) ∘ SE3(quat_k,pos_k) ∘ T(joint_pos_k).
    All our bodies have identity quat and only j6 has a nonzero joint_pos (and it's a
    leaf), so this reduces to jointPlacement_k = SE3(I, pos_k + joint_pos_k).
  - A site on body k sits at B_k ∘ SE3(euler, pos); relative to J_k that is
    T(-joint_pos_k) ∘ SE3(euler, pos).  This is what we hand to model.addFrame.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

# --- joint chain, VERBATIM from yam_{left,right}_body.xml (identical for both arms) ---
# (name, axis, body_pos rel parent, joint_pos in body, lower, upper). MJCF bodies carry
# NO rotation (identity quat); only j6 has a nonzero joint <pos> (the flange centre).
_CHAIN = [
    ("j1", (1.0, 0.0, 0.0),  (0.066479, -0.060421, -0.016708), (0.0, 0.0, 0.0),       0.0,       6.283185),
    ("j2", (0.0, 0.0, 1.0),  (0.0455,    0.02,     -0.0306),   (0.0, 0.0, 0.0),      -2.879793,  0.436332),
    ("j3", (0.0, 0.0, 1.0),  (1e-06,     0.2596,    0.06575),  (0.0, 0.0, 0.0),      -1.396263,  2.443461),
    ("j4", (0.0, 0.0, 1.0),  (0.06,      0.241549, -0.00205),  (0.0, 0.0, 0.0),      -1.570796,  1.570796),
    ("j5", (-1.0, 0.0, 0.0), (0.0403,    0.070162, -0.0339),   (0.0, 0.0, 0.0),      -1.570796,  1.570796),
    ("j6", (0.0, 1.0, 0.0),  (0.0,       0.0,       0.0),      (-0.0405, 0.0498, 0.0), -2.0944,   2.0944),
]

# Site placements: (parent joint index in _CHAIN, pos in that body's frame, euler xyz).
# wrist is on link4 (joint index 3); ee is on link6 (joint index 5). Identical wrist
# both sides; ee differs (ORCA flange). euler is MuJoCo intrinsic-xyz (only single-axis
# values occur here, so the order is immaterial).
_WRIST_SITE = (3, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
_EE_SITE = {
    # euler matches the MJCF literal "0 3.14159 0" verbatim (a truncated π) so the
    # Pinocchio model preserves the measured source transform.
    "left":  (5, (-0.0504, 0.1078, 0.0), (0.0, 3.14159, 0.0)),
    "right": (5, (-0.0306, 0.1078, 0.0), (0.0, 0.0,     0.0)),
}

JOINT_NAMES = lambda side: [f"{side}_arm_{n}" for (n, *_rest) in _CHAIN]  # noqa: E731


def joint_local_axis(suffix: str) -> np.ndarray:
    """Unit rotation axis (in the joint's own frame) for joint 'j1'..'j6'."""
    for name, axis, *_rest in _CHAIN:
        if name == suffix:
            a = np.asarray(axis, float)
            return a / np.linalg.norm(a)
    raise KeyError(suffix)


def _rot_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """MuJoCo intrinsic-xyz euler → rotation matrix (R = Rx·Ry·Rz)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def build_arm_model(side: str, max_vel: float = 12.0) -> pin.Model:
    """Build the standalone single-arm Pinocchio model (6 revolute joints) for `side`,
    with OP_FRAMEs `{side}_wrist` and `{side}_ee` at the MJCF site placements. All six
    joints get a finite velocityLimit so pink.VelocityLimit picks them up (the two-stage
    freeze then mutates model.velocityLimit in place)."""
    model = pin.Model()
    model.name = f"yam_{side}_arm"
    parent = 0  # 0 = universe
    jids: list[int] = []
    for name, axis, body_pos, joint_pos, lo, hi in _CHAIN:
        a = np.asarray(axis, float)
        a /= np.linalg.norm(a)
        jmodel = pin.JointModelRevoluteUnaligned(*a)  # arbitrary-axis hinge → axis verbatim
        placement = pin.SE3(np.eye(3), np.asarray(body_pos, float) + np.asarray(joint_pos, float))
        jid = model.addJoint(parent, jmodel, placement, f"{side}_arm_{name}")
        model.addJointFrame(jid)  # so child OP_FRAMEs have a parent joint frame
        iq, iv = model.joints[jid].idx_q, model.joints[jid].idx_v
        model.lowerPositionLimit[iq] = lo
        model.upperPositionLimit[iq] = hi
        model.velocityLimit[iv] = max_vel
        jids.append(jid)
        parent = jid

    def _add_site(name: str, spec) -> None:
        ji, pos, euler = spec
        jid = jids[ji]
        joint_pos = np.asarray(_CHAIN[ji][3], float)
        # site relative to the Pinocchio joint frame = T(-joint_pos) ∘ SE3(euler, pos)
        placement = pin.SE3(_rot_xyz(*euler), np.asarray(pos, float) - joint_pos)
        parent_frame = model.getFrameId(f"{side}_arm_{_CHAIN[ji][0]}")
        model.addFrame(pin.Frame(name, jid, parent_frame, placement, pin.FrameType.OP_FRAME))

    _add_site(f"{side}_wrist", _WRIST_SITE)
    _add_site(f"{side}_ee", _EE_SITE[side])
    return model
