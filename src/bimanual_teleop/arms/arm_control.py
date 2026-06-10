"""Per-arm controller: turns a tracked wrist pose into YAM joint targets.

Transport-agnostic — the sim loop and the (future) ZMQ arm process both just call
`update(hand_sample, engaged, t)`. Wraps ArmIK + ClutchMapper + One-Euro target
smoothing + a workspace bounding box. Holds the last pose when not engaged.
"""
from __future__ import annotations

import numpy as np

from ..filters import OneEuroFilter
from ..vr.calibrate import R_base_from_body
from ..vr.frames import SE3, ClutchMapper, HandSample, mat_to_se3, quat_to_R, r_base_from_vr
from .ik import ArmIK


class ArmController:
    def __init__(self, rig: dict, side: str):
        self.rig = rig
        self.side = side
        self.ik = ArmIK(rig, side)
        m = rig["mapping"]
        body_relative = bool(rig.get("vr", {}).get("body_relative", True))
        # The mapper R must match the frame the ctrl samples actually live in:
        # body axes (default) or legacy raw WebXR room coords. Built here, not
        # patched in later by the engine, because the absolute position mapping
        # depends on it from the very first engage.
        if body_relative:
            R = R_base_from_body(rig["arms"][side]["base_quat"])
        else:
            R = r_base_from_vr(rig["arms"][side]["base_quat"], m["r_base_from_vr_euler"][side])
        self.base_R = quat_to_R(rig["arms"][side]["base_quat"])   # base → world
        self.base_pos = np.asarray(rig["arms"][side]["base_pos"], dtype=float)
        # Absolute position mapping: operator torso→wrist lands 1:1 on robot
        # chest→wrist. The chest anchor defaults to the midpoint of the two arm
        # bases; absolute only makes sense for body-relative ctrl samples, so the
        # legacy raw-room mode falls back to clutch deltas.
        mode = str(m.get("position_mode", "absolute"))
        if not body_relative:
            mode = "relative"
        anchor_w = m.get("body_anchor_world")
        if anchor_w is None:
            # The base plates are the SHOULDER line; the operator's torso proxy is
            # sternum-ish, below their shoulders — and the YAM workspace, like a
            # human arm's, lives below the shoulder mounts. Drop the chest anchor
            # accordingly so torso-height hands land in reachable space.
            drop = float(m.get("body_anchor_drop", 0.15))
            anchor_w = 0.5 * (np.asarray(rig["arms"]["left"]["base_pos"], dtype=float)
                              + np.asarray(rig["arms"]["right"]["base_pos"], dtype=float)) \
                - np.array([0.0, 0.0, drop])
        chest_base = self.base_R.T @ (np.asarray(anchor_w, dtype=float) - self.base_pos)
        # Intrinsic wrist twist: pronation about YOUR forearm axis becomes a pure
        # j6 roll about the EE's own tool axis (never a j4/j5 swing through the
        # wrist singularity). Forced to 'world' in legacy raw-room mode.
        twist_mode = str(m.get("twist_mode", "intrinsic"))
        ori_mode = str(m.get("orientation_mode", "absolute"))
        if not body_relative:
            twist_mode = "world"
            ori_mode = "relative"
        C = None
        if ori_mode == "absolute":
            # Hand↔EE convention, derived (no calibration): the EE basis comes from
            # the frozen rest contract (fingers along approach, palm inward); the
            # hand basis from measured hand-local finger/palm-back axes.
            from ..viz.hand_geom import hand_basis_for_side
            B = hand_basis_for_side(self.ik, self.base_R, side)        # EE-local [lat|palm-back|fingers]
            f = np.asarray(m.get("hand_finger_axis", [0.345, -0.363, -0.866]), dtype=float)
            f = f / np.linalg.norm(f)
            p = np.asarray(m.get("hand_palm_axis", [-0.496, 0.713, -0.496]), dtype=float)
            p = p - (p @ f) * f
            p = p / np.linalg.norm(p)
            H = np.column_stack([np.cross(p, f), p, f])                # hand-local [lat|palm-back|fingers]
            C = H @ B.T                                                # EE-local → hand-local
        self.mapper = ClutchMapper(R, pos_scale=m["pos_scale"], position_mode=mode,
                                   chest_base=chest_base if mode == "absolute" else None,
                                   engage_blend_s=float(m.get("engage_blend_s", 1.0)),
                                   twist_mode=twist_mode,
                                   hand_twist_axis=m.get("hand_twist_axis", [0.0, 0.456, 0.890])
                                   if twist_mode == "intrinsic" else None,
                                   ee_tool_axis=self.ik.ee_tool_axis_local
                                   if twist_mode == "intrinsic" else None,
                                   orientation_mode=ori_mode, hand_ee_convention=C)
        # Anti-cross guard: keep this hand on its own side of the world Y axis so
        # the two arms can never overlap. left stays y ≤ -gap, right stays y ≥ +gap.
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
        self.cmd_pos = None  # last commanded EE position (base frame) after workspace/cross clamps

    def update(self, hand: HandSample | None, engaged: bool, t: float) -> np.ndarray:
        active = bool(engaged and hand is not None and hand.tracked)
        # (Re)anchor on the clutch rising edge OR whenever the mapper has dropped its
        # anchor while still active — set_R() calls release() (e.g. after legacy
        # calibration or manual mapper retuning), and without re-engaging here the next target()
        # would assert on a null anchor. engage() makes the target equal the current
        # EE pose, so re-anchoring is continuous (only the displacement-since-clutch
        # resets to zero, which is the desired behaviour after a retune).
        if active and (not self._was_engaged or not self.mapper.engaged):
            # anchor POSITION to the wrist site, ORIENTATION to the hand
            anchor = SE3.from_rotation_and_translation(
                self.ik.fk_ee().rotation(), self.ik.fk_wrist().translation())
            self.mapper.engage(mat_to_se3(hand.wrist), anchor, t)
            self.pos_filt = OneEuroFilter(**self._filt_params)   # reset smoothing on engage
        if not active and self._was_engaged:                 # release
            self.mapper.release()
        self._was_engaged = active

        if active:
            target = self.mapper.target(mat_to_se3(hand.wrist), t)
            p = np.clip(target.translation(), self.ws_min, self.ws_max)  # workspace box
            sm = self.pos_filt({"x": p[0], "y": p[1], "z": p[2]}, t)     # One-Euro
            pb = np.array([sm["x"], sm["y"], sm["z"]])                   # target in base frame
            pw = self.base_R @ pb + self.base_pos                        # → world
            pw[1] = min(pw[1], self.y_bound) if self.side == "left" else max(pw[1], self.y_bound)
            pb = self.base_R.T @ (pw - self.base_pos)                    # anti-cross clamp, back to base
            target = SE3.from_rotation_and_translation(target.rotation(), pb)
            self.cmd_pos = pb.copy()
            self.cmd_R = target.rotation().as_matrix()   # commanded orientation (base frame), for viz
            self.ik.solve(target, iters=self.iters)   # two-stage: position(j1-3) then orientation(j4-6)
        else:
            self.cmd_pos = None
            self.cmd_R = None
        return self.ik.q
