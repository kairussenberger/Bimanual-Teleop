"""Per-arm controller: turns a tracked wrist pose into YAM joint targets.

Transport-agnostic — the sim loop and the (future) ZMQ arm process both just call
`update(hand_sample, engaged, t)`. Wraps ArmIK + ClutchMapper + One-Euro target
smoothing + a workspace bounding box. Holds the last pose when not engaged.
"""
from __future__ import annotations

import numpy as np

import mink

from ..hands.retarget_core import OneEuroFilter
from ..hands.quest_retarget import hand_frame
from ..vr.frames import ClutchMapper, HandSample, mat_to_se3, quat_to_R, r_base_from_vr, rotvec
from .ik import ArmIK


class ArmController:
    def __init__(self, rig: dict, side: str):
        self.rig = rig
        self.side = side
        self.ik = ArmIK(rig, side)
        m = rig["mapping"]
        # Frame derived from THIS arm's real base orientation so "hand forward" →
        # "robot reaches forward" (not sideways). Calibration overrides this via
        # mapper.set_R(). tweak = optional per-side nudge.
        R = r_base_from_vr(rig["arms"][side]["base_quat"], m["r_base_from_vr_euler"][side])
        self.mapper = ClutchMapper(R, pos_scale=m["pos_scale"],
                                   abs_orientation=m.get("abs_orientation", True))
        # Anti-cross guard: keep this hand on its own side of the world Y axis so
        # the two arms can never overlap. left stays y ≤ -gap, right stays y ≥ +gap.
        self.base_R = quat_to_R(rig["arms"][side]["base_quat"])   # base → world
        self.base_pos = np.asarray(rig["arms"][side]["base_pos"], dtype=float)
        gap = float(rig.get("vr", {}).get("cross_gap", 0.05))
        self.y_bound = -gap if side == "left" else gap
        ws = rig["safety"]["workspace"]
        self.ws_min = np.asarray(ws["min"], dtype=float)
        self.ws_max = np.asarray(ws["max"], dtype=float)
        self.iters = int(rig["ik"].get("iters", 1))
        # lighter smoothing than the fingers -> less arm lag (snappier baseline,
        # beta cuts lag on fast motion). Tune if jittery.
        self._filt_params = dict(mincutoff=4.0, beta=1.0)
        self.pos_filt = OneEuroFilter(**self._filt_params)
        self.wrist_filt = OneEuroFilter(mincutoff=3.0, beta=0.6)
        self._was_engaged = False
        # Direct wrist-joint mapping (j4 pitch, j5 yaw, j6 roll). ref_frame is the
        # operator's hand frame at calibration; set via set_ref_frame().
        self.ref_frame: np.ndarray | None = None
        self.q0_wrist = np.asarray(rig["arms"][side]["neutral_q"][3:6], dtype=float)
        self.wrist_gain = float(rig["mapping"].get("wrist_gain", 1.0))
        self.wrist_signs = np.asarray(rig["mapping"].get("wrist_signs", {}).get(side, [1, 1, 1]), dtype=float)

    def set_ref_frame(self, R: np.ndarray) -> None:
        self.ref_frame = np.asarray(R, dtype=float).reshape(3, 3)

    def _wrist_joints(self, hand: HandSample, t: float) -> np.ndarray:
        """Map the operator's wrist rotation (vs calibration ref) directly to
        (j4,j5,j6): pitch→j4, yaw→j5, roll→j6. Returns home wrist if uncalibrated."""
        if self.ref_frame is None or hand is None or hand.landmarks is None:
            return self.q0_wrist
        cur = hand_frame(hand.landmarks)[1]                 # continuous hand frame (webxr)
        rv = rotvec(cur @ self.ref_frame.T)                 # wrist rotation since calibration
        x, y, z = self.ref_frame[:, 0], self.ref_frame[:, 1], self.ref_frame[:, 2]
        pitch, yaw, roll = float(rv @ x), float(rv @ z), float(rv @ y)   # about lateral/normal/forward
        tgt = self.q0_wrist + self.wrist_gain * self.wrist_signs * np.array([pitch, yaw, roll])
        sm = self.wrist_filt({"j4": tgt[0], "j5": tgt[1], "j6": tgt[2]}, t)
        return np.array([sm["j4"], sm["j5"], sm["j6"]])

    def update(self, hand: HandSample | None, engaged: bool, t: float) -> np.ndarray:
        active = bool(engaged and hand is not None and hand.tracked)
        if active and not self._was_engaged:                 # clutch rising edge
            self.mapper.engage(mat_to_se3(hand.wrist), self.ik.fk_ee())
            self.pos_filt = OneEuroFilter(**self._filt_params)   # reset smoothing on engage
        if not active and self._was_engaged:                 # release
            self.mapper.release()
        self._was_engaged = active

        if active:
            target = self.mapper.target(mat_to_se3(hand.wrist))
            p = np.clip(target.translation(), self.ws_min, self.ws_max)  # workspace box
            sm = self.pos_filt({"x": p[0], "y": p[1], "z": p[2]}, t)     # One-Euro
            pb = np.array([sm["x"], sm["y"], sm["z"]])                   # target in base frame
            pw = self.base_R @ pb + self.base_pos                        # → world
            pw[1] = min(pw[1], self.y_bound) if self.side == "left" else max(pw[1], self.y_bound)
            pb = self.base_R.T @ (pw - self.base_pos)                    # anti-cross clamp, back to base
            target = mink.SE3.from_rotation_and_translation(target.rotation(), pb)
            self.ik.set_wrist(self.q0_wrist)               # solve POSITION with the wrist at neutral...
            self.ik.solve(target, iters=self.iters)        # ...so the arm (j1-j3) never reacts to wrist motion
            q = self.ik.q
            q[3:6] = self._wrist_joints(hand, t)           # ...then overlay wrist orientation onto j4/j5/j6
            return q
        return self.ik.q
