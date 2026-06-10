"""The teleop engine: per-side arm + hand controllers.

The arm mapping is CALIBRATION-FREE in the default body-relative mode: positions
are torso→wrist vectors in body axes, and orientation is the wrist rotation since
clutch-engage mapped to the corresponding robot-world axes (see ClutchMapper).
`vr.calib_seconds` therefore defaults to 0 and the arms follow as soon as a hand
is tracked + engaged.

Setting `vr.calib_seconds > 0` enables the optional legacy startup STILLNESS hold
(operator stands arms at sides for N seconds). It measures the operator's body
frame from the head pose, which only steers arm motion when `vr.body_relative`
is explicitly disabled for diagnostics; otherwise it is a stillness/quality gate.

`calib_status` is published every tick for the in-headset visual (vr/vuer_source):
the operator can SEE the countdown and when to stop holding still.

Sink = anything with set_arm(side, q) / set_hand(side, joints_deg): Unity render
now, the hardware drivers later.
"""
from __future__ import annotations

from .arms.arm_control import ArmController
from .config import SIDES
from .hands.hand_control import HandController
from .vr.calibrate import Calibrator, R_base_from_body, body_relative_hand_sample
from .vr.frames import HandSample, VRFrame


class TeleopEngine:
    def __init__(self, rig: dict, sink):
        self.rig = rig
        self.sink = sink
        self.arm = {s: ArmController(rig, s) for s in SIDES}
        self.hand = {s: HandController(rig, s) for s in SIDES}
        self.calibrator = Calibrator(rig)
        self.calib_s = float(rig.get("vr", {}).get("calib_seconds", 0.0))
        self.calibrated = self.calib_s <= 0
        self.body_relative = bool(rig.get("vr", {}).get("body_relative", True))
        self.torso_from_head = rig.get("vr", {}).get("torso_from_head", [0.0, -0.35, 0.0])
        self._calib_t0 = None
        self._prompted = False
        self._done_t = None
        # Published each tick for the in-headset visual. None once the post-calib
        # banner has faded. See vr/vuer_source.set_calib.
        self.calib_status: dict | None = None
        if not self.calibrated:
            self.calib_status = {"active": True, "phase": "wait", "progress": 0.0,
                                 "remaining": self.calib_s, "left": False, "right": False,
                                 "msg": "DROP YOUR ARMS TO YOUR SIDES"}
        elif self.body_relative:
            self._set_body_relative_mapping()

    def tick(self, frame: VRFrame | None, engaged: dict[str, bool], t: float) -> None:
        if not self.calibrated:
            self._calibration_tick(frame, t)
            return
        # Keep the "CALIBRATED ✓" banner up briefly, then clear it.
        if self.calib_status is not None:
            if self._done_t is not None and (t - self._done_t) > 2.5:
                self.calib_status = None
        for s in SIDES:
            hs = frame.hands.get(s) if frame else None
            self.sink.set_arm(s, self.arm[s].update(self._arm_hand_sample(hs, frame), engaged.get(s, False), t))
            self.sink.set_hand(s, self.hand[s].update(hs, t))

    def _set_body_relative_mapping(self) -> None:
        for s in SIDES:
            self.arm[s].mapper.set_R(R_base_from_body(self.rig["arms"][s]["base_quat"]))

    def _arm_hand_sample(self, hs: HandSample | None, frame: VRFrame | None) -> HandSample | None:
        if not self.body_relative:
            return hs
        return body_relative_hand_sample(hs, frame.head if frame else None, self.torso_from_head)

    def _calibration_tick(self, frame: VRFrame | None, t: float) -> None:
        """Collect resting-stance samples; hold arms at the rest pose; fingers track."""
        head = frame.head if frame else None
        seen = {s: False for s in SIDES}
        for s in SIDES:
            hs = frame.hands.get(s) if frame else None
            if hs and hs.tracked and hs.landmarks is not None:
                self.calibrator.add(s, hs.landmarks, hs.wrist, head)
                seen[s] = True
            self.sink.set_arm(s, self.arm[s].ik.q0)            # hold rest pose
            self.sink.set_hand(s, self.hand[s].update(hs, t))  # fingers can track meanwhile
        any_tracked = seen["left"] or seen["right"]

        if any_tracked and self._calib_t0 is None:
            self._calib_t0 = t
            if not self._prompted:
                print("[calib] ARMS DOWN at your sides, relaxed, palms rolled INWARD — "
                      f"matching the robot's rest. Hold still {self.calib_s:.0f}s...", flush=True)
                self._prompted = True

        # Publish status for the in-headset visual.
        if self._calib_t0 is None:
            self.calib_status = {"active": True, "phase": "wait", "progress": 0.0,
                                 "remaining": self.calib_s, "left": seen["left"],
                                 "right": seen["right"], "msg": "DROP YOUR ARMS TO YOUR SIDES"}
        else:
            elapsed = t - self._calib_t0
            self.calib_status = {"active": True, "phase": "hold",
                                 "progress": max(0.0, min(1.0, elapsed / self.calib_s)),
                                 "remaining": max(0.0, self.calib_s - elapsed),
                                 "left": seen["left"], "right": seen["right"],
                                 "msg": "HOLD STILL — ARMS AT YOUR SIDES"}

        if self._calib_t0 is not None and (t - self._calib_t0) >= self.calib_s:
            allok = True
            for s in SIDES:
                r = self.calibrator.result(s)
                if r is None:
                    print(f"[calib] {s}: NOT tracked — using default frame.", flush=True)
                    allok = False
                    continue
                self.arm[s].mapper.set_R(R_base_from_body(self.rig["arms"][s]["base_quat"])
                                         if self.body_relative else r["R"])
                allok = allok and r["ok"]
                tag = "OK" if r["ok"] else "SHAKY (hold stiller & recalibrate)"
                print(f"[calib] {s}: {tag} | stillness={r['std']*1000:.0f}mm "
                      f"| forward≈{r['forward'].round(2)} up≈{r['up'].round(2)}", flush=True)
            print("[calib] done — arms now follow your hands. (Restart to recalibrate.)", flush=True)
            self._done_t = t
            self.calib_status = {"active": True, "phase": "done", "progress": 1.0,
                                 "remaining": 0.0, "left": seen["left"], "right": seen["right"],
                                 "msg": "CALIBRATED" if allok else "CALIBRATED (shaky — recheck)"}
            self.calibrated = True
