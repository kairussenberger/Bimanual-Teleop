"""HardwareSink: the real-robot backend behind the same set_arm/set_hand interface
as RenderSink, so TeleopEngine drives visualization or hardware unchanged.

NOTE: this is the synchronous, single-process bring-up sink. For production the
arms want a dedicated ~250 Hz CAN loop per side (separate process / SCHED_FIFO),
decoupled from vision/IK via latest-value buffers — see README "Hardware day" and
the recon architecture. This class is the correct *logic*; wrap each arm in its
own process when you need the rate.
"""
from __future__ import annotations

import numpy as np

from .config import SIDES


class HardwareSink:
    def __init__(self, rig: dict):
        from .arms.yam_driver import YamArm
        from .hands.real_driver import RealHand
        self.arms = {s: YamArm(rig["arms"][s]["can_channel"]) for s in SIDES}
        self.hands = {s: RealHand(model_name=rig["hands"][s]["model_name"]) for s in SIDES}

    def set_arm(self, side: str, q: np.ndarray) -> None:
        self.arms[side].command(q)

    def set_hand(self, side: str, joints_deg: dict) -> None:
        self.hands[side].set_joint_positions(joints_deg)

    def close(self) -> None:
        for h in self.hands.values():
            try:
                h.release()
            except Exception:
                pass
        for a in self.arms.values():
            try:
                a.close()
            except Exception:
                pass
