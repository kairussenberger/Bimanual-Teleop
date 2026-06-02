"""The teleop engine: per-side arm + hand controllers, with a startup CALIBRATION
phase. On connect the operator holds both hands out front (palms down, fingers
spread) for `vr.calib_seconds`; we measure their hand frame and align each arm's
headset→base rotation to it. During calibration the arms hold their home pose
(fingers may still track). After it, arms follow the operator.

Sink = anything with set_arm(side, q) / set_hand(side, joints_deg): the sim world
now, the hardware drivers later.
"""
from __future__ import annotations

from .arms.arm_control import ArmController
from .config import SIDES
from .hands.hand_control import HandController
from .vr.calibrate import Calibrator
from .vr.frames import VRFrame


class TeleopEngine:
    def __init__(self, rig: dict, sink):
        self.rig = rig
        self.sink = sink
        self.arm = {s: ArmController(rig, s) for s in SIDES}
        self.hand = {s: HandController(rig, s) for s in SIDES}
        self.calibrator = Calibrator(rig)
        self.calib_s = float(rig.get("vr", {}).get("calib_seconds", 5.0))
        self.calibrated = self.calib_s <= 0
        self._calib_t0 = None
        self._prompted = False

    def tick(self, frame: VRFrame | None, engaged: dict[str, bool], t: float) -> None:
        if not self.calibrated:
            self._calibration_tick(frame, t)
            return
        for s in SIDES:
            hs = frame.hands.get(s) if frame else None
            self.sink.set_arm(s, self.arm[s].update(hs, engaged.get(s, False), t))
            self.sink.set_hand(s, self.hand[s].update(hs, t))

    def _calibration_tick(self, frame: VRFrame | None, t: float) -> None:
        """Collect reference-stance samples; hold arms at home; fingers track."""
        any_tracked = False
        for s in SIDES:
            hs = frame.hands.get(s) if frame else None
            if hs and hs.tracked and hs.landmarks is not None:
                self.calibrator.add(s, hs.landmarks)
                any_tracked = True
            self.sink.set_arm(s, self.arm[s].ik.q0)            # hold home
            self.sink.set_hand(s, self.hand[s].update(hs, t))  # fingers can track meanwhile
        if any_tracked and self._calib_t0 is None:
            self._calib_t0 = t
            if not self._prompted:
                print("[calib] HOLD both hands out in front, palms DOWN, fingers SPREAD — "
                      f"measuring your frame for {self.calib_s:.0f}s...", flush=True)
                self._prompted = True
        if self._calib_t0 is not None and (t - self._calib_t0) >= self.calib_s:
            applied = []
            for s in SIDES:
                R = self.calibrator.compute(s)
                if R is not None:
                    self.arm[s].mapper.set_R(R)                     # position frame
                    self.arm[s].set_ref_frame(self.calibrator.ref_frame(s))  # wrist-rotation zero
                    applied.append(s)
            print(f"[calib] done (calibrated: {applied or 'none — using defaults'}). "
                  "Arms now follow your hands.", flush=True)
            self.calibrated = True
