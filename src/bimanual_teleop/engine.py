"""The teleop engine: holds the per-side arm + hand controllers and applies one
tick (VRFrame + per-side engage flags → arm joints + finger joints) onto a robot
sink. The sink is the sim world now and the hardware drivers later — both expose
set_arm(side, q) and set_hand(side, joints_deg).
"""
from __future__ import annotations

from .arms.arm_control import ArmController
from .config import SIDES
from .hands.hand_control import HandController
from .vr.frames import VRFrame


class TeleopEngine:
    def __init__(self, rig: dict, sink):
        self.rig = rig
        self.sink = sink                       # anything with set_arm/set_hand
        self.arm = {s: ArmController(rig, s) for s in SIDES}
        self.hand = {s: HandController(rig, s) for s in SIDES}

    def tick(self, frame: VRFrame | None, engaged: dict[str, bool], t: float) -> None:
        for s in SIDES:
            hs = frame.hands.get(s) if frame else None
            self.sink.set_arm(s, self.arm[s].update(hs, engaged.get(s, False), t))
            self.sink.set_hand(s, self.hand[s].update(hs, t))
