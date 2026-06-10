"""Stylized ORCA hand geometry, articulated by the REAL 17 joint angles.

orca_core ships motors and configs but no meshes, so renderers draw this
parametric hand instead: a palm slab plus five fingers (2 box segments each,
thumb 2 + root offset), flexed by the live mcp/pip(/thumb) angles and splayed by
the abd angles — real grasp data, stylized geometry. If official ORCA meshes ever
become available, swap them in here.

Frames are NOT guessed: the canonical hand frame (x = lateral toward thumb,
y = back-of-hand normal, z = finger direction) is registered onto the EE-local
frame per side at import of the rig, using the frozen rest contract — approach
axis points world-DOWN at rest, palms face the body midline ("palms rolled
inward", docs/RESTING_POSE.md).
"""
from __future__ import annotations

import numpy as np

# canonical proportions (m) — eyeballed from the ORCA hand's rough envelope
_PALM_W, _PALM_L, _PALM_T = 0.080, 0.085, 0.024
_FINGERS = ("index", "middle", "ring", "pinky")
_F_X = {"index": 0.028, "middle": 0.009, "ring": -0.010, "pinky": -0.029}
_SEG = {"index": (0.042, 0.034), "middle": (0.046, 0.037),
        "ring": (0.042, 0.034), "pinky": (0.034, 0.027)}
_F_W = 0.015


def _box(p0: np.ndarray, p1: np.ndarray, lat: np.ndarray, width: float, thick: float) -> np.ndarray:
    """(12,3,3) triangle box from p0 to p1, cross-section width×thick."""
    axis = p1 - p0
    n = np.linalg.norm(axis)
    if n < 1e-9:
        axis, n = np.array([0.0, 0.0, 1e-6]), 1e-6
    a = axis / n
    u = lat - (lat @ a) * a
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(a, u)
    du, dv = 0.5 * width * u, 0.5 * thick * v
    c = [p0 - du - dv, p0 + du - dv, p0 + du + dv, p0 - du + dv,
         p1 - du - dv, p1 + du - dv, p1 + du + dv, p1 - du + dv]
    quads = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    tris = []
    for q0, q1, q2, q3 in quads:
        tris.append([c[q0], c[q1], c[q2]])
        tris.append([c[q0], c[q2], c[q3]])
    return np.asarray(tris, dtype=float)


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def orca_hand_tris_canonical(joints_deg: dict, mirror: bool = False) -> np.ndarray:
    """(n,3,3) triangles in the CANONICAL hand frame (x lateral-to-thumb, y back
    of hand, z fingers), articulated by ORCA joint degrees (0 = open; mcp/pip
    flex toward the palm at −y; abd splays about y). `mirror` flips x for the
    other chirality."""
    g = lambda k: float(joints_deg.get(k, 0.0))
    parts = [_box(np.array([0.0, 0.0, 0.005]), np.array([0.0, 0.0, _PALM_L]),
                  np.array([1.0, 0.0, 0.0]), _PALM_W, _PALM_T)]
    for f in _FINGERS:
        root = np.array([_F_X[f], 0.0, _PALM_L])
        l1, l2 = _SEG[f]
        R = _rot_y(np.radians(g(f"{f}_abd"))) @ _rot_x(-np.radians(g(f"{f}_mcp")))
        k1 = root + R @ np.array([0.0, 0.0, l1])
        parts.append(_box(root, k1, R @ np.array([1.0, 0.0, 0.0]), _F_W, _F_W * 0.85))
        R2 = R @ _rot_x(-np.radians(g(f"{f}_pip")))
        k2 = k1 + R2 @ np.array([0.0, 0.0, l2])
        parts.append(_box(k1, k2, R2 @ np.array([1.0, 0.0, 0.0]), _F_W * 0.9, _F_W * 0.75))
    # thumb: root on the lateral palm edge, angled out, cmc+abd orient, mcp+dip curl
    t_root = np.array([0.5 * _PALM_W, -0.2 * _PALM_T, 0.030])
    Rt = (_rot_y(np.radians(-35.0)) @ _rot_x(-np.radians(0.6 * g("thumb_cmc")))
          @ _rot_y(-np.radians(0.5 * g("thumb_abd"))))
    t1 = t_root + Rt @ np.array([0.0, 0.0, 0.046])
    parts.append(_box(t_root, t1, Rt @ np.array([1.0, 0.0, 0.0]), _F_W * 1.15, _F_W))
    Rt2 = Rt @ _rot_x(-np.radians(g("thumb_mcp") + 0.7 * g("thumb_dip")))
    t2 = t1 + Rt2 @ np.array([0.0, 0.0, 0.036])
    parts.append(_box(t1, t2, Rt2 @ np.array([1.0, 0.0, 0.0]), _F_W, _F_W * 0.8))
    tris = np.concatenate(parts, axis=0)
    if mirror:
        tris = tris[:, ::-1, :].copy()          # flip winding with the reflection
        tris[:, :, 0] *= -1.0
    return tris


def hand_basis_for_side(ik, base_R: np.ndarray, side: str) -> np.ndarray:
    """EE-local 3×3 mapping canonical hand axes → EE frame, registered from the
    frozen rest contract: fingers along the approach axis pointing world-DOWN at
    rest; palm face (−y canonical) toward the body midline (palms rolled inward).
    Columns = where canonical x (lateral), y (back of hand), z (fingers) land."""
    import pinocchio as pin
    data = ik.model.createData()
    pin.forwardKinematics(ik.model, data, ik.q0)
    pin.updateFramePlacements(ik.model, data)
    ee_R_base = np.asarray(data.oMf[ik.model.getFrameId(f"{ik.side}_ee")].rotation)
    ee_R_world = base_R @ ee_R_base
    a = ik.ee_tool_axis_local.copy()
    if (ee_R_world @ a)[2] > 0:                    # approach must point DOWN at rest
        a = -a
    inward = np.array([0.0, 1.0, 0.0]) if side == "left" else np.array([0.0, -1.0, 0.0])
    inward_local = ee_R_world.T @ inward           # palm-face target, in EE-local
    pn = -(inward_local - (inward_local @ a) * a)  # back of hand = OPPOSITE the palm face
    pn = pn / (np.linalg.norm(pn) + 1e-12)
    lat = np.cross(pn, a)                          # x = y × z (right-handed canonical)
    return np.column_stack([lat, pn, a])


def orca_hand_tris_ee(joints_deg: dict, basis: np.ndarray, mirror: bool = False) -> np.ndarray:
    """Articulated hand triangles in the EE-LOCAL frame (attach with the EE pose)."""
    return orca_hand_tris_canonical(joints_deg, mirror=mirror) @ basis.T
