"""Per-arm controller: turns a tracked wrist pose into YAM joint targets.

Transport-agnostic — the sim loop and the (future) ZMQ arm process both just call
`update(hand_sample, engaged, t)`. Wraps ArmIK + ClutchMapper + One-Euro target
smoothing + a workspace bounding box. Holds the last pose when not engaged.
"""
from __future__ import annotations

import numpy as np

from ..hands.retarget_core import OneEuroFilter
from ..vr.frames import ClutchMapper, HandSample, mat_to_se3, r_base_from_vr
from .ik import ArmIK


class ArmController:
    def __init__(self, rig: dict, side: str):
        self.rig = rig
        self.side = side
        self.ik = ArmIK(rig, side)
        m = rig["mapping"]
        # Frame derived from THIS arm's real base orientation so "hand forward" →
        # "robot reaches forward" (not sideways). tweak = optional per-side nudge.
        R = r_base_from_vr(rig["arms"][side]["base_quat"], m["r_base_from_vr_euler"][side])
        self.mapper = ClutchMapper(R, pos_scale=m["pos_scale"], abs_orientation=False)
        ws = rig["safety"]["workspace"]
        self.ws_min = np.asarray(ws["min"], dtype=float)
        self.ws_max = np.asarray(ws["max"], dtype=float)
        self.iters = int(rig["ik"].get("iters", 1))
        # lighter smoothing than the fingers -> less arm lag (snappier baseline,
        # beta cuts lag on fast motion). Tune if jittery.
        self._filt_params = dict(mincutoff=4.0, beta=1.0)
        self.pos_filt = OneEuroFilter(**self._filt_params)
        self._was_engaged = False

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
            import mink
            target = mink.SE3.from_rotation_and_translation(
                target.rotation(), np.array([sm["x"], sm["y"], sm["z"]]))
            self.ik.solve(target, iters=self.iters)
        return self.ik.q
