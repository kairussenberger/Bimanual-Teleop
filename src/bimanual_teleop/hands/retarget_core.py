"""Pure, input-agnostic finger-retargeting primitives.

Ported from orca-teleop/webcam_teleop.py (the proven MediaPipe pipeline) so the
VR path can reuse the *exact* geometry → ORCA-joint-degrees math and the One-Euro
smoothing without dragging in MediaPipe/OpenCV. The only change vs. the original
is that `landmarks_to_joint_angles` takes an explicit `abd_sign` and drops the
calib.json / pinch-snap coupling (those can be layered back on later).

Output contract (matches webcam_teleop): a dict {orca_joint: degrees}, started
from `neutral`, overwriting the subset of the 17 ORCA joints we can estimate.
"""
from __future__ import annotations

import numpy as np

# --- landmark layout (MediaPipe-style 21-point hand) ----------------------- #
FINGERS = ["index", "middle", "ring", "pinky"]
LM = {  # (base, pip, dip, tip) indices into a 21-point hand
    "thumb":  (1, 2, 3, 4),   # (cmc, mcp, ip, tip)
    "index":  (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "pinky":  (17, 18, 19, 20),
}
WRIST_LM = 0
MIDDLE_BASE_LM = 9

# --- flexion normalization (rad) and output degree ranges (from webcam_teleop) #
MCP_STRAIGHT, MCP_CURLED = 2.95, 1.45
PIP_STRAIGHT, PIP_CURLED = 2.90, 0.70
THUMB_STRAIGHT, THUMB_CURLED = 2.90, 1.30
MCP_OPEN, MCP_CLOSE = 0.0, 95.0
PIP_OPEN, PIP_CLOSE = 0.0, 100.0
THUMB_MCP_OPEN, THUMB_MCP_CLOSE = 0.0, 80.0
THUMB_DIP_OPEN, THUMB_DIP_CLOSE = 0.0, 85.0
THUMB_ABD_MIN_ANG, THUMB_ABD_MAX_ANG = 18.0, 70.0
ABD_GAIN = 1.0

ONE_EURO_MINCUTOFF = 1.7
ONE_EURO_BETA = 0.30


# --- geometry helpers ------------------------------------------------------ #
def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle (rad) at vertex b between segments b->a and b->c."""
    ba, bc = a - b, c - b
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba < 1e-8 or nbc < 1e-8:
        return float(np.pi)
    return float(np.arccos(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)))


def flex_fraction(angle: float, straight: float, curled: float) -> float:
    """0.0 straight → 1.0 fully curled."""
    return float(np.clip((straight - angle) / (straight - curled), 0.0, 1.0))


def signed_angle_2d(ref: np.ndarray, v: np.ndarray) -> float:
    """Signed angle (rad) from 2D vector ref to v (CCW positive)."""
    ref = ref / (np.linalg.norm(ref) + 1e-8)
    v = v / (np.linalg.norm(v) + 1e-8)
    dot = np.clip(np.dot(ref, v), -1.0, 1.0)
    cross = ref[0] * v[1] - ref[1] * v[0]
    return float(np.arctan2(cross, dot))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp_to_rom(angles: dict, roms: dict) -> dict:
    """Clamp each joint to roms[j]=[lo,hi]; drop joints absent from roms."""
    out = {}
    for j, v in angles.items():
        if j in roms:
            lo, hi = roms[j]
            out[j] = float(np.clip(v, lo, hi))
    return out


class OneEuroFilter:
    """One-Euro adaptive low-pass over a dict of scalar channels (Casiez 2012).

    Smooths hard when steady (kills jitter), barely smooths during fast motion
    (kills lag). One instance per hand; state persists across frames. Verbatim
    from webcam_teleop.py.
    """

    def __init__(self, mincutoff: float = ONE_EURO_MINCUTOFF,
                 beta: float = ONE_EURO_BETA, dcutoff: float = 1.0):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self._x_prev: dict = {}
        self._dx_prev: dict = {}
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, values: dict, t: float) -> dict:
        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = dict(values)
            self._dx_prev = {k: 0.0 for k in values}
            return dict(values)
        dt = max(t - self._t_prev, 1e-3)
        self._t_prev = t
        out = {}
        for k, x in values.items():
            x_prev = self._x_prev.get(k, x)
            dx = (x - x_prev) / dt
            a_d = self._alpha(self.dcutoff, dt)
            dx_hat = a_d * dx + (1 - a_d) * self._dx_prev.get(k, 0.0)
            cutoff = self.mincutoff + self.beta * abs(dx_hat)
            a = self._alpha(cutoff, dt)
            x_hat = a * x + (1 - a) * x_prev
            self._x_prev[k], self._dx_prev[k] = x_hat, dx_hat
            out[k] = x_hat
        return out


def landmarks_to_joint_angles(pts: np.ndarray, neutral: dict, *, mirror: bool = True,
                              use_wrist: bool = False) -> dict:
    """Geometric retarget of a 21-point hand → {orca_joint: degrees}.

    `pts` is (21, 3). For VR we pass points already expressed in a hand-LOCAL
    frame (x across palm, y wrist→middle, z palm normal), so the 2D abduction
    math (which uses x,y) is meaningful. Flexion uses 3D interior angles and is
    frame-invariant. Mirrors webcam_teleop.landmarks_to_joint_angles (minus the
    calib.json/pinch coupling).
    """
    pts = np.asarray(pts, dtype=float)
    wrist = pts[WRIST_LM]
    out = dict(neutral)
    palm_axis_2d = (pts[MIDDLE_BASE_LM] - wrist)[:2]
    abd_sign = -1.0 if mirror else 1.0

    for f in FINGERS:
        base, pip, dip, _ = LM[f]
        f_mcp = flex_fraction(joint_angle(wrist, pts[base], pts[pip]), MCP_STRAIGHT, MCP_CURLED)
        f_pip = flex_fraction(joint_angle(pts[base], pts[pip], pts[dip]), PIP_STRAIGHT, PIP_CURLED)
        out[f"{f}_mcp"] = lerp(MCP_OPEN, MCP_CLOSE, f_mcp)
        out[f"{f}_pip"] = lerp(PIP_OPEN, PIP_CLOSE, f_pip)
        prox_dir_2d = (pts[pip] - pts[base])[:2]
        spread = abd_sign * np.rad2deg(signed_angle_2d(palm_axis_2d, prox_dir_2d)) * ABD_GAIN
        ext = 1.0 - f_mcp  # trust spread only while the finger is extended
        out[f"{f}_abd"] = lerp(neutral.get(f"{f}_abd", 0.0), float(spread), ext)

    cmc, mcp, ip, tip = LM["thumb"]
    out["thumb_mcp"] = lerp(THUMB_MCP_OPEN, THUMB_MCP_CLOSE,
                            flex_fraction(joint_angle(pts[cmc], pts[mcp], pts[ip]),
                                          THUMB_STRAIGHT, THUMB_CURLED))
    out["thumb_dip"] = lerp(THUMB_DIP_OPEN, THUMB_DIP_CLOSE,
                            flex_fraction(joint_angle(pts[mcp], pts[ip], pts[tip]),
                                          THUMB_STRAIGHT, THUMB_CURLED))
    thumb_meta = pts[mcp] - pts[cmc]
    index_meta = pts[LM["index"][0]] - wrist
    cos = np.dot(thumb_meta, index_meta) / (
        np.linalg.norm(thumb_meta) * np.linalg.norm(index_meta) + 1e-8)
    abd_ang = np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0)))
    abd_frac = float(np.clip((abd_ang - THUMB_ABD_MIN_ANG)
                             / (THUMB_ABD_MAX_ANG - THUMB_ABD_MIN_ANG), 0.0, 1.0))
    out["thumb_abd"] = lerp(5.0, 55.0, abd_frac)
    out["thumb_cmc"] = neutral.get("thumb_cmc", 0.0)

    if use_wrist:
        palm = pts[MIDDLE_BASE_LM] - wrist
        out["wrist"] = float(np.clip(np.rad2deg(np.arctan2(palm[1], -palm[2] - 1e-6)) * 0.5, -60, 30))
    return out
