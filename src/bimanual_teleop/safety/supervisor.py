"""Safety supervisor: the single place that decides who may move.

It consumes the latest VRFrame + a clutch + an e-stop flag and produces, per side,
whether the arm/hand should follow the operator, applying:
- a staleness gate (VR sample older than safety.staleness_s ⇒ not fresh),
- VR-loss handling (engaged but stale ⇒ HOLD for safety.hold_s, then drop to IDLE
  so the arm doesn't chase a frozen target or snap),
- a latched ESTOP (zeros engagement until reset()).

In sim this runs inline in the control loop; on hardware the same logic runs in a
dedicated supervisor process that also owns torque enable/disable. It is clock-
injected (pass `t`) so it's deterministic and testable.
"""
from __future__ import annotations

from ..config import SIDES
from ..vr.frames import VRFrame
from .clutch import AlwaysOn, Clutch
from .states import State


class Supervisor:
    def __init__(self, rig: dict, clutch: Clutch | None = None):
        s = rig["safety"]
        self.staleness_s = float(s["staleness_s"])
        self.hold_s = float(s["hold_s"])
        self.clutch = clutch or AlwaysOn()
        self.state = State.DISCONNECTED
        self._estop = False
        self._last_fresh_t = {side: -1e9 for side in SIDES}
        self._engaged_since = {side: None for side in SIDES}

    def estop(self) -> None:
        self._estop = True
        self.state = State.ESTOP

    def reset(self) -> None:
        self._estop = False
        self.state = State.IDLE

    def update(self, frame: VRFrame | None, t: float) -> dict[str, bool]:
        """Return {side: engaged}. Updates self.state for display/telemetry."""
        if self._estop:
            self.state = State.ESTOP
            return {side: False for side in SIDES}

        fresh_frame = frame is not None and (t - frame.stamp) <= self.staleness_s
        engaged = {}
        any_engaged = False
        for side in SIDES:
            want = self.clutch.engaged(side, frame)
            tracked = frame is not None and side in frame.hands and frame.hands[side].tracked
            if fresh_frame and tracked:
                self._last_fresh_t[side] = t
            stale_for = t - self._last_fresh_t[side]

            if want and fresh_frame and tracked:
                eng = True
            elif self._engaged_since[side] is not None and stale_for <= self.hold_s:
                eng = True   # brief HOLD across a tracking dropout (don't snap)
            else:
                eng = False

            self._engaged_since[side] = (self._engaged_since[side] or t) if eng else None
            engaged[side] = eng
            any_engaged = any_engaged or eng

        if frame is None:
            self.state = State.DISCONNECTED
        elif any_engaged:
            self.state = State.ENGAGED
        else:
            self.state = State.IDLE
        return engaged
