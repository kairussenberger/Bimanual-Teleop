"""Operator-frame calibration.

The user holds both hands out in front, palms down, fingers spread, for a few
seconds. From the (averaged) hand landmarks we measure the operator's body axes
in the WebXR frame — forward (fingertips), up (back-of-hand normal), right — and
solve the rotation that maps them onto the robot's world axes (robot faces world
−X, +Z up). Per side, that rotation is then expressed in the arm's IK base frame.

This replaces guessed frame constants with a measurement of how the operator
actually holds their hands, so "hand forward → arm forward" is correct for each
hand regardless of headset orientation.
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


def operator_axes(lm: np.ndarray) -> np.ndarray:
    """(right, up, forward) operator axes (columns) in the WebXR frame, from a
    reference-stance hand: fingertips give forward, the back-of-hand normal gives
    up (disambiguated by WebXR +Y = away from gravity)."""
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


def calibrate_R(lm_avg: np.ndarray, base_quat) -> np.ndarray:
    """R_base_from_vr for one arm: maps a WebXR wrist displacement into the arm's
    IK base frame so the measured operator axes align with the robot world axes."""
    Op = operator_axes(lm_avg)                       # operator axes in WebXR frame
    R_world_from_vr = W_AXES @ Op.T                  # WebXR → world
    return quat_to_R(base_quat).T @ R_world_from_vr  # world → base


class Calibrator:
    def __init__(self, rig: dict):
        self.rig = rig
        self._samples: dict[str, list] = {"left": [], "right": []}

    def add(self, side: str, landmarks) -> None:
        if landmarks is not None:
            self._samples[side].append(np.asarray(landmarks, dtype=float).reshape(25, 3))

    def count(self, side: str) -> int:
        return len(self._samples[side])

    def result(self, side: str) -> dict | None:
        """Rigorous calibration for one side over the most-settled sample window.
        Returns {R, ref, ok, std, forward, up} or None if too few samples. `ok` is
        False when the hand wasn't held still enough (high variance) → re-calibrate."""
        from ..hands.quest_retarget import hand_frame
        s = self._samples[side]
        if len(s) < 8:
            return None
        arr = np.stack(s[-30:])                                   # most-settled window
        avg = arr.mean(axis=0)
        fwd = arr[:, [W_INDEX_TIP, W_MID_TIP, W_RING_TIP], :].mean(1) - arr[:, W_WRIST, :]
        std = float(np.linalg.norm(fwd.std(axis=0)))              # how still the hand was held
        R = calibrate_R(avg, self.rig["arms"][side]["base_quat"])
        Op = operator_axes(avg)
        ok = std < 0.02 and bool(np.isfinite(R).all())
        return {"R": R, "ref": hand_frame(avg)[1], "ok": ok, "std": std,
                "forward": Op[:, 2], "up": Op[:, 1]}

    # Back-compat thin wrappers
    def compute(self, side: str) -> np.ndarray | None:
        r = self.result(side)
        return r["R"] if r else None

    def ref_frame(self, side: str) -> np.ndarray | None:
        r = self.result(side)
        return r["ref"] if r else None
