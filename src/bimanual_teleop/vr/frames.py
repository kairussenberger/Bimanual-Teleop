"""VR pose types + the relative/clutch SE(3) mapping from a tracked wrist to an
arm end-effector target.

Mapping (per OpenTeleVision / Quest2ROS best practice): on clutch *engage* we
latch the current wrist pose and the current EE pose as anchors. While engaged,
the EE target is the anchored EE pose composed with the operator's wrist motion
*relative* to its anchor — translation scaled and rotated into the arm base frame
by a constant `R_base_from_vr`, orientation either absolute-aligned or relative.
Because it's relative, absolute origin offsets cancel, so frame calibration only
needs that one rotation (mirrored per arm about the sagittal plane).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mink
import numpy as np


@dataclass
class HandSample:
    tracked: bool = False
    wrist: np.ndarray = field(default_factory=lambda: np.eye(4))  # 4x4 in headset/world frame
    landmarks: np.ndarray | None = None                          # (25,3) WebXR joints
    pinch: float = 0.0                                           # 0..1 pinch strength


@dataclass
class VRFrame:
    stamp: float = 0.0
    head: np.ndarray = field(default_factory=lambda: np.eye(4))
    hands: dict[str, HandSample] = field(default_factory=dict)


def quat_to_R(q) -> np.ndarray:
    """Quaternion (w, x, y, z) → 3x3 rotation matrix (MuJoCo quat convention)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


# WebXR reference frame (x=right, y=up, -z=forward) → robot WORLD frame
# (x, y, z; the robot faces world -X). Columns = where webxr +x,+y,+z land in world:
# webxr +z (back) → world +X  (so forward -z → -X = robot forward),
# webxr +y (up)   → world +Z,  webxr +x (right) → world +Y.
WEBXR_TO_WORLD = np.array([[0.0, 0.0, 1.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])


def r_base_from_vr(base_quat, tweak_euler=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Rotation mapping a wrist displacement in the WebXR frame to a displacement
    in this arm's IK base frame, so 'hand forward' → 'robot reaches forward'.

    The arm's standalone IK base frame is rotated into the world by `base_quat`,
    so: R = (world←base)ᵀ · (webxr→world) · tweak  =  base_quatᵀ · WEBXR_TO_WORLD · tweak.
    `tweak_euler` is an optional small correction (radians) applied in the WebXR
    frame if an axis still reads inverted."""
    R_bw = quat_to_R(base_quat)                 # base → world
    return R_bw.T @ WEBXR_TO_WORLD @ euler_to_R(tweak_euler)


def euler_to_R(euler_xyz) -> np.ndarray:
    """Intrinsic XYZ euler (rad) → 3x3 rotation (matches MuJoCo eulerseq XYZ)."""
    cx, cy, cz = np.cos(euler_xyz)
    sx, sy, sz = np.sin(euler_xyz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → rotation vector (axis * angle)."""
    R = np.asarray(R, dtype=float)
    ang = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if ang < 1e-7:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return ang / (2.0 * np.sin(ang)) * v


def mat_to_se3(T: np.ndarray) -> mink.SE3:
    T = np.asarray(T, dtype=float).reshape(4, 4)
    return mink.SE3.from_rotation_and_translation(
        mink.SO3.from_matrix(T[:3, :3]), T[:3, 3])


class ClutchMapper:
    """Relative+clutch wrist→EE mapping for one arm."""

    def __init__(self, R_base_from_vr: np.ndarray, pos_scale: float = 1.0,
                 abs_orientation: bool = True):
        self.R = np.asarray(R_base_from_vr, dtype=float).reshape(3, 3)
        self.scale = float(pos_scale)
        self.abs_orientation = bool(abs_orientation)
        self.anchor_ctrl: mink.SE3 | None = None
        self.anchor_ee: mink.SE3 | None = None
        self._R_off: np.ndarray | None = None   # orientation anchor so engage is continuous
        # Orientation correspondence (hand-LOCAL → EE-LOCAL). Calibration sets this so
        # a wrist twist maps to the EE roll axis (j6). Identity = no remap.
        self.P = np.eye(3)
        # Debug: when True, the EE holds the anchor orientation and ONLY position
        # maps — lets you verify the 3 translation axes in isolation before adding
        # orientation. Set via the --pos-only studio flag.
        self.freeze_ori = False

    def set_R(self, R: np.ndarray) -> None:
        """Replace the headset→base rotation (e.g. after calibration)."""
        self.R = np.asarray(R, dtype=float).reshape(3, 3)
        self.release()   # force a fresh anchor on next engage

    def set_P(self, P: np.ndarray) -> None:
        """Replace the hand-local→EE-local orientation correspondence (calibration)."""
        self.P = np.asarray(P, dtype=float).reshape(3, 3)
        self.release()

    @property
    def engaged(self) -> bool:
        return self.anchor_ctrl is not None

    def engage(self, ctrl: mink.SE3, ee: mink.SE3) -> None:
        """Latch position AND orientation anchors on the clutch rising edge, so the
        target equals the current EE pose at the engage instant (no jump)."""
        self.anchor_ctrl = ctrl
        self.anchor_ee = ee
        # abs mode: Rt = R_off @ (R @ ctrl.R); choose R_off so Rt == anchor_ee.R at engage.
        self._R_off = ee.rotation().as_matrix() @ (self.R @ ctrl.rotation().as_matrix()).T

    def release(self) -> None:
        self.anchor_ctrl = None
        self.anchor_ee = None
        self._R_off = None

    def target(self, ctrl: mink.SE3) -> mink.SE3:
        """EE target (arm base frame) for the current wrist pose while engaged."""
        assert self.anchor_ctrl is not None and self.anchor_ee is not None
        dp_vr = ctrl.translation() - self.anchor_ctrl.translation()
        p = self.anchor_ee.translation() + self.scale * (self.R @ dp_vr)
        if self.freeze_ori:                       # position-only debug: hold rest orientation
            return mink.SE3.from_rotation_and_translation(self.anchor_ee.rotation(), p)
        # Orientation: the operator's wrist rotation SINCE engage, measured in the
        # hand-local frame, re-expressed into the EE frame via the calibrated
        # correspondence P, then applied to the EE anchor. Continuous at engage
        # (dR=I ⇒ Rt=anchor_ee), and a twist about the hand's pointing axis maps to
        # the EE roll axis (j6) because P aligns the two frames. The old "absolute"
        # form let R cancel (RᵀR=I), so calibration couldn't steer orientation at all.
        dR = self.anchor_ctrl.rotation().inverse().as_matrix() @ ctrl.rotation().as_matrix()
        Rt = self.anchor_ee.rotation().as_matrix() @ (self.P @ dR @ self.P.T)
        return mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(Rt), p)
