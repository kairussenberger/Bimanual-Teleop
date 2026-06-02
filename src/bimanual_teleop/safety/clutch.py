"""Pluggable clutch / deadman. With hand-tracking only (no controllers), "follow
me" must be an explicit, held intent. A clutch decides, per side, whether the
operator wants that arm to follow *this* frame.

Implementations:
- AlwaysOn: sim/fake default (both sides always engaged).
- GestureClutch: engage a side while that hand holds a gesture that the grasp
  doesn't use — default is a sustained open-palm spread (all fingers extended).
  Re-grip to re-anchor. Tune the gesture to taste; a bluetooth foot pedal is the
  ergonomic ideal and can implement this same interface.
- KeyboardClutch: a shared flag an external key handler toggles (viewer/CLI).
"""
from __future__ import annotations

from ..vr.frames import VRFrame


class Clutch:
    def engaged(self, side: str, frame: VRFrame | None) -> bool:  # pragma: no cover
        raise NotImplementedError


class AlwaysOn(Clutch):
    def engaged(self, side: str, frame: VRFrame | None) -> bool:
        return frame is not None and side in frame.hands and frame.hands[side].tracked


class GestureClutch(Clutch):
    """Engage `side` while that hand's pinch is RELEASED (open) — i.e. fingers
    extended. Pinching then commands a grasp without disengaging the arm. This is
    a placeholder convention; swap for a dedicated engage gesture or a pedal."""

    def __init__(self, release_below: float = 0.3):
        self.release_below = release_below

    def engaged(self, side: str, frame: VRFrame | None) -> bool:
        if frame is None or side not in frame.hands:
            return False
        h = frame.hands[side]
        return h.tracked  # always track; refine with a dedicated engage gesture


class KeyboardClutch(Clutch):
    """Engaged follows an externally-set flag (e.g. a viewer key callback sets
    .held = True while a key is pressed). Acts as a co-pilot deadman in sim."""

    def __init__(self):
        self.held = False

    def engaged(self, side: str, frame: VRFrame | None) -> bool:
        tracked = frame is not None and side in frame.hands and frame.hands[side].tracked
        return self.held and tracked
