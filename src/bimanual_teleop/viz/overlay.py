"""Shared MuJoCo scene-overlay primitives: RGB frame triads, arrows, point-to-
point connectors, spheres, and a 25-joint WebXR hand skeleton.

Every function APPENDS geoms to an MjvScene — either the passive viewer's
`v.user_scn` or a `mujoco.Renderer().scene` (offscreen). All guard against the
geom buffer overflowing. Used by launch/run_sim.py (on-screen frame overlay) and
tools/mapping_studio.py (side-by-side operator↔robot viz) so "how we draw a
frame" lives in one place.
"""
from __future__ import annotations

import mujoco
import numpy as np

# WebXR 25-joint hand bone topology (W3C XRHand order): wrist(0) → each finger's
# metacarpal → proximal → … → tip. Matches hands/quest_retarget.py's indices.
HAND_BONES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),                    # thumb
    (0, 5), (5, 6), (6, 7), (7, 8), (8, 9),            # index
    (0, 10), (10, 11), (11, 12), (12, 13), (13, 14),   # middle
    (0, 15), (15, 16), (16, 17), (17, 18), (18, 19),   # ring
    (0, 20), (20, 21), (21, 22), (22, 23), (23, 24),   # pinky
)

# X=red, Y=green, Z=blue (matches the prior run_sim.py triad colors exactly so the
# refactor is behavior-preserving).
AXIS_RGB = ((1.0, 0.2, 0.2), (0.2, 1.0, 0.2), (0.35, 0.35, 1.0))


def _alloc(scn):
    """Reserve the next geom slot, or None if the buffer is full."""
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    scn.ngeom += 1
    return g


def sphere(scn, pos, radius: float, rgba) -> None:
    g = _alloc(scn)
    if g is None:
        return
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, radius, radius]), np.asarray(pos, float),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))


def connector(scn, p0, p1, width: float, rgba) -> None:
    """A capsule spanning p0→p1 (used for skeleton bones)."""
    g = _alloc(scn)
    if g is None:
        return
    # init sets the color; mjv_connector then overwrites pos/size/orientation.
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.eye(3).flatten(), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         np.asarray(p0, float), np.asarray(p1, float))


def arrow(scn, pos, direction, length: float, width: float, rgba) -> None:
    """An arrow at `pos` pointing along `direction` (a MuJoCo arrow points along
    its local +Z, so we build a frame whose Z is the requested direction)."""
    d = np.asarray(direction, float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return
    g = _alloc(scn)
    if g is None:
        return
    z = d / n
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    mat = np.column_stack([x, y, z]).flatten()
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW,
                        np.array([width, width, length]), np.asarray(pos, float),
                        mat, np.asarray(rgba, np.float32))


def triad(scn, pos, R, length: float = 0.12, width: float = 0.007, alpha: float = 1.0) -> None:
    """An RGB axis triad (columns of R) at `pos`. X=red, Y=green, Z=blue."""
    R = np.asarray(R, float).reshape(3, 3)
    for i in range(3):
        c = AXIS_RGB[i]
        arrow(scn, pos, R[:, i], length, width, (c[0], c[1], c[2], alpha))


def skeleton(scn, joints, width: float = 0.004, rgba=(0.95, 0.85, 0.2, 1.0),
             joint_radius: float = 0.006) -> None:
    """Draw a (≥25,3) WebXR hand as bones + joint spheres (world coordinates)."""
    j = np.asarray(joints, float).reshape(-1, 3)
    for a, b in HAND_BONES:
        if a < len(j) and b < len(j):
            connector(scn, j[a], j[b], width, rgba)
    for p in j:
        sphere(scn, p, joint_radius, rgba)
