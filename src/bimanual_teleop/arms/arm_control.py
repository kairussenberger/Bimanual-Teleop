"""Per-arm controller: turns a tracked wrist pose into YAM joint targets.

Transport-agnostic — the sim loop and the (future) ZMQ arm process both just call
`update(hand_sample, engaged, t)`. Wraps ArmIK + ClutchMapper + One-Euro target
smoothing + a workspace bounding box. Holds the last pose when not engaged.
"""
from __future__ import annotations

import numpy as np

import mink

from ..filters import OneEuroFilter
from ..vr.frames import ClutchMapper, HandSample, euler_to_R, mat_to_se3, quat_to_R, r_base_from_vr
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
        self._was_engaged = False
        self.cmd_R = None   # last commanded EE orientation (base frame) — for the on-screen viz

    def set_ref_frame(self, R: np.ndarray) -> None:
        # Retained for API compatibility; orientation is handled by the two-stage
        # IK against the calibrated mapper target, not a hand-rolled decomposition.
        pass

    def set_ori_calib(self, wrist_ref: np.ndarray | None, op_axes: np.ndarray | None) -> None:
        """Build the hand-local→EE-local orientation correspondence P from the
        reference stance, so a wrist TWIST drives the EE roll joint (j6) instead of
        arcing the forearm (j4). op_axes = operator [right|up|forward] in WebXR;
        wrist_ref = the WebXR wrist rotation at the reference stance."""
        if wrist_ref is None or op_axes is None:
            return
        L = np.asarray(wrist_ref, float).T @ np.asarray(op_axes, float)  # operator axes in hand-local
        E_loc = self.ik.ee_semantic_frame_local()                        # EE axes in EE-local
        # Optional per-side correspondence nudge (radians, intrinsic XYZ in hand-local).
        # Stays a proper rotation, so it can re-align an axis but never re-mirror. Use
        # e.g. [0, π, 0] if wrist roll still feels reversed for a side after calibration.
        tweak = (self.rig.get("mapping", {}).get("ori_tweak_euler", {}) or {}).get(self.side, (0.0, 0.0, 0.0))
        self.mapper.set_P(E_loc @ L.T @ euler_to_R(tweak))               # hand-local → EE-local

    def update(self, hand: HandSample | None, engaged: bool, t: float) -> np.ndarray:
        active = bool(engaged and hand is not None and hand.tracked)
        # (Re)anchor on the clutch rising edge OR whenever the mapper has dropped its
        # anchor while still active — set_R()/set_P() call release() (e.g. live frame
        # retuning in mapping_studio), and without re-engaging here the next target()
        # would assert on a null anchor. engage() makes the target equal the current
        # EE pose, so re-anchoring is continuous (only the displacement-since-clutch
        # resets to zero, which is the desired behaviour after a retune).
        if active and (not self._was_engaged or not self.mapper.engaged):
            # anchor POSITION to the wrist site, ORIENTATION to the hand
            anchor = mink.SE3.from_rotation_and_translation(
                self.ik.fk_ee().rotation(), self.ik.fk_wrist().translation())
            self.mapper.engage(mat_to_se3(hand.wrist), anchor)
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
            self.cmd_R = target.rotation().as_matrix()   # commanded orientation (base frame), for viz
            self.ik.solve(target, iters=self.iters)   # two-stage: position(j1-3) then orientation(j4-6)
        else:
            self.cmd_R = None
        return self.ik.q
