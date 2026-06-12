"""HardwareSink: the real-robot backend behind the same set_arm/set_hand interface
as RenderSink, so TeleopEngine drives visualization or hardware unchanged.

Every arm command passes through a per-side JointCommandShaper before touching
CAN: clamped to the physical joint limits, per-joint speed-capped
(`hardware.rate_limit`), acceleration-capped (`hardware.accel_limit` — velocity
ramps instead of slamming, so the physical frame is never jerked), and smoothed
by a critically-damped tracker (`hardware.smooth_hz`) feeding the YAM's
motor-side MIT PD. All caps are per second of wall-clock time, so the motion is
the same whatever rate the loop achieves. The shaper initializes from the arm's
MEASURED pose, so the first command glides from wherever the robot actually is
— no startup snap.

NOTE: this is the synchronous, single-process bring-up sink. For production the
arms want a dedicated ~250 Hz CAN loop per side (separate process / SCHED_FIFO),
decoupled from vision/IK via latest-value buffers — see README "Hardware day" and
the recon architecture. This class is the correct *logic*; wrap each arm in its
own process when you need the rate.
"""
from __future__ import annotations

import time

import numpy as np

from .config import SIDES
from .logging_utils import get_logger
from .safety.shaper import JointCommandShaper

log = get_logger("hardware")


def arm_shaper(rig: dict, q0) -> JointCommandShaper:
    """The hardware-boundary shaper for one YAM arm, from rig config. Factored out
    so the safety wiring is unit-testable without the i2rt SDK."""
    hw = rig.get("hardware", {})
    limits = rig["arms"]["joint_limits"]
    return JointCommandShaper(
        q0,
        rate_limit=float(hw.get("rate_limit", 1.2)),
        smooth_hz=float(hw.get("smooth_hz", 3.0)),
        accel_limit=float(hw.get("accel_limit", 12.0)),
        lo=limits["lower"],
        hi=limits["upper"],
    )


class HardwareSink:
    def __init__(self, rig: dict):
        from .arms.yam_driver import YamArm
        from .hands.real_driver import RealHand
        self.arms = {s: YamArm(rig["arms"][s]["can_channel"]) for s in SIDES}
        self.hands = {s: RealHand(model_name=rig["hands"][s]["model_name"]) for s in SIDES}
        self.shapers = {}
        for s in SIDES:
            try:
                q0 = self.arms[s].state()                  # glide from the MEASURED pose
            except Exception as e:
                q0 = np.asarray(rig["arms"][s]["neutral_q"], dtype=float)
                log.warning("%s arm: could not read measured pose (%s); shaper starts at rig neutral", s, e)
            self.shapers[s] = arm_shaper(rig, q0)

    def set_arm(self, side: str, q: np.ndarray) -> None:
        self.arms[side].command(self.shapers[side].shape(q, time.monotonic()))

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
