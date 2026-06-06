"""Operator-frame calibration — RESTING-POSE variant.

The operator stands with both arms hanging at their sides, relaxed, palms rolled
inward toward their body — mirroring the robot's frozen ragdoll-hang `neutral_q`
(see docs/RESTING_POSE.md). We hold that for `vr.calib_seconds` and measure the
correspondence "operator at rest ↔ robot at rest", from which arm motion proceeds.

Key change from the old palms-down-forward stance: the operator's BODY axes come
from the HEAD pose (where you're facing + gravity), NOT from hand geometry. With
hands at your sides the fingers point down, so the old fingertips-as-forward
measurement was degenerate — it corrupted both the position frame R and the wrist
orientation correspondence P (the flipped-wrist bug). The head is reliable in any
hand pose. The wrist's resting orientation still gives the per-hand reference.
"""
from __future__ import annotations

import numpy as np

from .frames import quat_to_R

# WebXR 25-joint indices (W3C order)
W_WRIST = 0
W_INDEX_PROX, W_INDEX_TIP = 6, 9
W_MID_TIP = 14
W_RING_TIP = 19
W_PINKY_PROX = 21

# Desired robot WORLD axes for the reference stance: operator right → +Y,
# operator up → +Z, operator forward → −X (the robot faces −X).
W_AXES = np.column_stack([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])

# WebXR gravity-up (the headset floor frame is +Y up).
_UP = np.array([0.0, 1.0, 0.0])


def head_op_axes(head_mat: np.ndarray) -> np.ndarray:
    """(right, up, forward) operator BODY axes (columns) in the WebXR frame, from
    the head pose. Forward = the way you're facing, flattened to horizontal; up =
    gravity. Robust regardless of where the hands are (works with arms at sides).

    The WebXR camera looks down its local −Z, so view-forward = −(head R)·ẑ."""
    H = np.asarray(head_mat, dtype=float).reshape(4, 4)[:3, :3]
    fwd = -H[:, 2]                                   # camera −Z = view forward
    fwd = fwd - (fwd @ _UP) * _UP                    # flatten to horizontal
    n = np.linalg.norm(fwd)
    if n < 1e-6:                                     # looking straight up/down → fall back to raw
        fwd = -H[:, 2]
        n = np.linalg.norm(fwd) + 1e-9
    f = fwd / n
    r = np.cross(f, _UP); r /= (np.linalg.norm(r) + 1e-9)   # operator right
    u = np.cross(r, f); u /= (np.linalg.norm(u) + 1e-9)
    return np.column_stack([r, u, f])


def operator_axes(lm: np.ndarray) -> np.ndarray:
    """(right, up, forward) operator axes (columns) in the WebXR frame from a
    palms-down-FORWARD hand (legacy fallback only, when no head pose is available).
    Fingertips give forward, the back-of-hand normal gives up."""
    lm = np.asarray(lm, dtype=float).reshape(25, 3)
    wrist = lm[W_WRIST]
    fwd = lm[[W_INDEX_TIP, W_MID_TIP, W_RING_TIP]].mean(0) - wrist     # fingertips → forward
    palm_lat = lm[W_INDEX_PROX] - lm[W_PINKY_PROX]                     # across the palm
    n = np.cross(fwd, palm_lat)
    if n[1] < 0:                                                       # up points away from gravity
        n = -n
    f = fwd / (np.linalg.norm(fwd) + 1e-9)
    r = np.cross(f, n); r /= (np.linalg.norm(r) + 1e-9)               # right = forward × up
    u = np.cross(r, f); u /= (np.linalg.norm(u) + 1e-9)
    return np.column_stack([r, u, f])


def _avg_rotation(mats: list) -> np.ndarray | None:
    """Average a list of rotation matrices, re-projected onto SO(3) via SVD."""
    if not mats:
        return None
    M = np.mean(np.stack(mats), axis=0)
    U, _, Vt = np.linalg.svd(M)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:                 # guard against a reflection
        U[:, -1] *= -1
        Rm = U @ Vt
    return Rm


def R_base_from_op(op_axes: np.ndarray, base_quat) -> np.ndarray:
    """R_base_from_vr for one arm: maps a WebXR wrist displacement into the arm's
    IK base frame so the measured operator axes align with the robot world axes."""
    R_world_from_vr = W_AXES @ np.asarray(op_axes, float).T   # WebXR → world
    return quat_to_R(base_quat).T @ R_world_from_vr           # world → base


def calibrate_R(lm_avg: np.ndarray, base_quat) -> np.ndarray:
    """Legacy hand-stance R (kept for back-compat / fallback)."""
    return R_base_from_op(operator_axes(lm_avg), base_quat)


class Calibrator:
    def __init__(self, rig: dict):
        self.rig = rig
        self._samples: dict[str, list] = {"left": [], "right": []}
        self._wrists: dict[str, list] = {"left": [], "right": []}   # wrist rotation 3x3
        self._wrist_pos: dict[str, list] = {"left": [], "right": []}  # wrist translation
        self._heads: list = []                                       # head 3x3, shared

    def add(self, side: str, landmarks, wrist=None, head=None) -> None:
        if landmarks is not None:
            self._samples[side].append(np.asarray(landmarks, dtype=float).reshape(25, 3))
        if wrist is not None:
            w = np.asarray(wrist, dtype=float).reshape(4, 4)
            self._wrists[side].append(w[:3, :3])
            self._wrist_pos[side].append(w[:3, 3])
        if head is not None:
            self._heads.append(np.asarray(head, dtype=float).reshape(4, 4)[:3, :3])

    def count(self, side: str) -> int:
        return len(self._wrists[side]) or len(self._samples[side])

    def tracked(self, side: str) -> bool:
        return self.count(side) > 0

    def _op_axes(self, side: str):
        """Operator body axes — head-derived (robust at any hand pose). Falls back to
        the legacy hand-stance measurement only if no head pose was seen."""
        if self._heads:
            H = _avg_rotation(self._heads[-30:])
            T = np.eye(4); T[:3, :3] = H
            return head_op_axes(T)
        if self._samples[side]:
            print(f"[WARN] calibrate: no head pose for {side}; falling back to hand-stance "
                  "axes (expects palms-down-forward, NOT arms-at-sides).", flush=True)
            return operator_axes(np.stack(self._samples[side][-30:]).mean(0))
        return None

    def result(self, side: str) -> dict | None:
        """Calibration for one side from the at-rest window. Returns
        {R, ref, op_axes, wrist_ref, ok, std, forward, up} or None if too few samples.
        `ok` is False when the hand wasn't held still enough → re-calibrate."""
        w = self._wrists[side]
        if len(w) < 8:
            return None
        op = self._op_axes(side)
        if op is None:
            return None
        base_quat = self.rig["arms"][side]["base_quat"]
        R = R_base_from_op(op, base_quat)
        wrist_ref = _avg_rotation(w[-30:])
        # Stillness from wrist TRANSLATION jitter over the window (fingertips-forward
        # is meaningless with the hand at the side, so don't use it).
        pos = np.stack(self._wrist_pos[side][-30:]) if self._wrist_pos[side] else None
        std = float(np.linalg.norm(pos.std(axis=0))) if pos is not None and len(pos) > 1 else 0.0
        ok = std < 0.02 and bool(np.isfinite(R).all()) and wrist_ref is not None
        return {"R": R, "ref": None, "op_axes": op, "wrist_ref": wrist_ref,
                "ok": ok, "std": std, "forward": op[:, 2], "up": op[:, 1]}

    # Back-compat thin wrappers
    def compute(self, side: str) -> np.ndarray | None:
        r = self.result(side)
        return r["R"] if r else None

    def ref_frame(self, side: str) -> np.ndarray | None:
        r = self.result(side)
        return r["ref"] if r else None
