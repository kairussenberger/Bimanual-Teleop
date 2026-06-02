"""Per-hand controller: turns a Quest hand skeleton into ORCA joint targets (deg).

Transport-agnostic. Wraps quest_to_orca + One-Euro smoothing + ROM clamp, started
from the hand's config neutral. Holds neutral when the hand isn't tracked.
"""
from __future__ import annotations

import time

from ..vr.frames import HandSample
from . import retarget_core as rc
from .joint_map import load_hand_config
from .quest_retarget import quest_to_orca


class HandController:
    def __init__(self, rig: dict, side: str):
        self.side = side
        self.neutral, self.roms = load_hand_config(rig["hands"][side]["model_name"])
        self.mirror = bool(rig["mapping"]["abd_mirror"][side])
        self.filt = rc.OneEuroFilter()
        self.last = dict(self.neutral)

    def update(self, hand: HandSample | None, t: float | None = None) -> dict:
        if hand is None or not hand.tracked or hand.landmarks is None:
            return self.last
        t = time.time() if t is None else t
        raw = rc.clamp_to_rom(quest_to_orca(hand.landmarks, self.neutral, mirror=self.mirror), self.roms)
        self.last = self.filt({**self.neutral, **raw}, t)
        return self.last
