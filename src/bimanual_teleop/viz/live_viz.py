"""Local 3D teleop viewer on Rerun — see what the engine is doing WITHOUT Unity,
a headset, or hardware. Works with every VR source (fake/synthetic/orbit/replay);
with `--vr replay` the Rerun timeline scrubs the whole session.

Scene (robot WORLD frame, +Z up, −X forward):
  world/arm/<side>/links     live link polyline base→j1..j6→EE (from Pinocchio FK)
  world/arm/<side>/ee        achieved EE triad
  world/arm/<side>/cmd       commanded EE target triad (only while engaged)
  world/op/<side>/wrist      operator torso→wrist arrow + wrist triad, mapped into
                             robot world by the SAME body→world axes the arm
                             mapping uses — if the robot disagrees with this
                             arrow/triad, the mapping (not the IK) is wrong
  err/<side>/*               cmd-vs-achieved position (cm) / orientation (deg)
  op/<side>/pinch, engaged   clutch debugging scalars
  status                     engagement / tracking / calibration text log

Requires the optional dependency:  uv sync --extra telemetry
"""
from __future__ import annotations

import numpy as np

from ..config import SIDES
from ..vr.calibrate import W_AXES, body_relative_hand_sample
from ..vr.frames import quat_to_R, rotvec
from .rerun_log import RerunLogger

_SIDE_RGB = {"left": (80, 170, 255), "right": (255, 120, 80)}


class TeleopViz:
    """Per-tick Rerun scene writer. Construct once, call tick() in the loop."""

    def __init__(self, rig: dict, *, spawn: bool = True, save_path: str | None = None,
                 max_hz: float = 30.0):
        self.log = RerunLogger("bimanual_teleop", spawn=spawn, save_path=save_path)
        if not self.log.enabled:
            raise SystemExit(
                "--viz needs the Rerun SDK (optional extra). Install it with:\n"
                "    uv sync --extra telemetry")
        self.rig = rig
        self.torso_from_head = np.asarray(rig.get("vr", {}).get("torso_from_head",
                                                                [0.0, -0.35, 0.0]), float)
        self.base_R = {s: quat_to_R(rig["arms"][s]["base_quat"]) for s in SIDES}
        self.base_p = {s: np.asarray(rig["arms"][s]["base_pos"], float) for s in SIDES}
        # Operator overlay anchor: the robot's "chest" (midpoint of the arm bases).
        self.torso_anchor = 0.5 * (self.base_p["left"] + self.base_p["right"])
        self._min_dt = 1.0 / float(max_hz)
        self._last_t = None
        self._last_status = None
        self._statics()

    def _statics(self) -> None:
        self.log.triad("world/axes", [0.0, 0.0, 0.0], np.eye(3), length=0.25)
        for s in SIDES:
            self.log.triad(f"world/arm/{s}/base", self.base_p[s], self.base_R[s], length=0.08)

    # ---- helpers ------------------------------------------------------------- #
    def _to_world(self, side: str, pts: np.ndarray) -> np.ndarray:
        return (self.base_R[side] @ np.atleast_2d(pts).T).T + self.base_p[side]

    # ---- per-tick ------------------------------------------------------------ #
    def tick(self, engine, frame, engaged: dict[str, bool], hz: float, t: float) -> None:
        if self._last_t is not None and (t - self._last_t) < self._min_dt:
            return
        self._last_t = t
        self.log.set_time(t)
        self.log.scalar("rate/loop_hz", hz)

        for s in SIDES:
            arm = engine.arm[s]
            self.log.linestrip(f"world/arm/{s}/links", self._to_world(s, arm.ik.link_points()),
                               color=_SIDE_RGB[s])
            ee = arm.ik.fk_ee()
            ee_p = self._to_world(s, ee.translation())[0]
            ee_R = self.base_R[s] @ ee.rotation().as_matrix()
            self.log.triad(f"world/arm/{s}/ee", ee_p, ee_R, length=0.09)

            if arm.cmd_pos is not None and arm.cmd_R is not None:
                cmd_p = self._to_world(s, arm.cmd_pos)[0]
                cmd_R = self.base_R[s] @ arm.cmd_R
                self.log.triad(f"world/arm/{s}/cmd", cmd_p, cmd_R, length=0.06)
                self.log.scalar(f"err/{s}/pos_cm", 100.0 * float(np.linalg.norm(cmd_p - ee_p)))
                self.log.scalar(f"err/{s}/ori_deg",
                                float(np.degrees(np.linalg.norm(rotvec(ee_R.T @ cmd_R)))))
            else:
                self.log.clear(f"world/arm/{s}/cmd")

            hs = frame.hands.get(s) if frame else None
            body = body_relative_hand_sample(hs, frame.head if frame else None,
                                             self.torso_from_head)
            if body is not None and body.tracked:
                wrist_w = self.torso_anchor + W_AXES @ body.wrist[:3, 3]
                self.log.arrow(f"world/op/{s}/vec", self.torso_anchor,
                               wrist_w - self.torso_anchor, color=_SIDE_RGB[s])
                # W_AXES and the body basis each carry one reflection; the product is
                # the proper rotation the arm mapping uses — same axes, same handedness.
                self.log.triad(f"world/op/{s}/wrist", wrist_w, W_AXES @ body.wrist[:3, :3],
                               length=0.07)
            else:
                self.log.clear(f"world/op/{s}/vec")
                self.log.clear(f"world/op/{s}/wrist")
            self.log.scalar(f"op/{s}/pinch", float(hs.pinch) if hs is not None else 0.0)
            self.log.scalar(f"op/{s}/engaged", 1.0 if engaged.get(s) else 0.0)

        status = (tuple(sorted((s, bool(engaged.get(s))) for s in SIDES)),
                  None if engine.calib_status is None else engine.calib_status.get("msg"))
        if status != self._last_status:
            self._last_status = status
            calib = engine.calib_status
            msg = " | ".join(f"{s}:{'ENGAGED' if engaged.get(s) else 'off'}" for s in SIDES)
            if calib is not None:
                msg += f" | calib: {calib.get('msg', '')}"
            self.log.text("status", msg)
