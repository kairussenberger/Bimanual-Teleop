"""Quest WebXR hand-tracking → ORCA finger joints (degrees).

WebXR exposes 25 joints per hand (W3C hand-input order). We remap those to the
21 MediaPipe-equivalent points, express them in a hand-LOCAL frame (so the 2D
abduction math is meaningful), and reuse the proven geometric retarget from
retarget_core. This bypasses MediaPipe entirely while reusing its tuned mapping.

Also provides `synthetic_webxr_hand(curl)` to drive the pipeline without a headset.
"""
from __future__ import annotations

import numpy as np

from . import retarget_core as rc

# W3C WebXR hand joint indices (per hand):
#  0 wrist
#  1-4   thumb:  metacarpal, phalanx-proximal, phalanx-distal, tip
#  5-9   index:  metacarpal, proximal, intermediate, distal, tip
#  10-14 middle, 15-19 ring, 20-24 pinky (same 5-joint layout)
# MediaPipe-equivalent 21 points = [wrist, thumb(cmc,mcp,ip,tip), then per finger
# (mcp,pip,dip,tip)]. We take each finger's proximal..tip (skip the metacarpal).
WEBXR_TO_MP = np.array([0,
                        1, 2, 3, 4,          # thumb metacarpal/proximal/distal/tip
                        6, 7, 8, 9,          # index proximal..tip
                        11, 12, 13, 14,      # middle
                        16, 17, 18, 19,      # ring
                        21, 22, 23, 24])     # pinky
_W = {"index": 6, "middle": 11, "ring": 16, "pinky": 21, "thumb": 1}  # proximal/metacarpal index


def hand_frame(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hand-local frame from a (25,3) WebXR skeleton: origin at wrist,
    y = wrist→middle-proximal, x = pinky→index across the palm, z = palm normal.
    Returns (origin, R) with R's columns the local axes in world coords."""
    origin = w[0]
    y = w[_W["middle"]] - origin
    x = w[_W["index"]] - w[_W["pinky"]]
    y = y / (np.linalg.norm(y) + 1e-9)
    x = x - np.dot(x, y) * y
    x = x / (np.linalg.norm(x) + 1e-9)
    z = np.cross(x, y)
    R = np.column_stack([x, y, z])
    return origin, R


def quest_to_orca(webxr25: np.ndarray, neutral: dict, *, mirror: bool) -> dict:
    """(25,3) WebXR hand → {orca_joint: degrees}, started from `neutral`."""
    w = np.asarray(webxr25, dtype=float).reshape(25, 3)
    origin, R = hand_frame(w)
    local = (w - origin) @ R           # express all joints in the hand-local frame
    pts21 = local[WEBXR_TO_MP]
    return rc.landmarks_to_joint_angles(pts21, neutral, mirror=mirror)


# --------------------------------------------------------------------------- #
# Synthetic hand (no headset): straight at curl=0, fist at curl=1.
# --------------------------------------------------------------------------- #
_FINGER_X = {"index": 0.02, "middle": 0.0, "ring": -0.02, "pinky": -0.04}  # palm spread (m)
_SEG = 0.035  # phalanx length (m)


def synthetic_webxr_hand(curl: float, *, thumb_curl: float | None = None) -> np.ndarray:
    """A crude but geometrically-valid (25,3) WebXR hand for testing the retarget.
    Palm in the x-y plane, fingers extend +y, curl bends phalanges toward -z."""
    curl = float(np.clip(curl, 0.0, 1.0))
    tc = curl if thumb_curl is None else float(np.clip(thumb_curl, 0.0, 1.0))
    w = np.zeros((25, 3))
    bend = curl * 1.4  # rad per joint at full curl

    def finger(start_idx, x0, n_seg, base_y, c):
        p = np.array([x0, base_y, 0.0])
        w[start_idx] = p                       # metacarpal / first joint
        ang = 0.0
        d = np.array([0.0, 1.0, 0.0])
        for k in range(1, n_seg):
            ang += (c * 1.4) if k >= 1 else 0.0
            d = np.array([0.0, np.cos(ang), -np.sin(ang)])
            p = p + _SEG * d
            w[start_idx + k] = p

    # thumb: 4 joints, offset to +x, curls toward palm
    tp = np.array([0.03, 0.01, 0.0])
    w[1] = tp
    ang = 0.0
    for k in range(1, 4):
        ang += tc * 1.1
        tp = tp + _SEG * np.array([np.cos(ang) * 0.3, np.sin(0.6), -np.sin(ang)])
        w[1 + k] = tp
    # four fingers: 5 joints each (metacarpal + 4)
    for name, base in (("index", 5), ("middle", 10), ("ring", 15), ("pinky", 20)):
        finger(base, _FINGER_X[name], 5, 0.0, curl)
    return w
