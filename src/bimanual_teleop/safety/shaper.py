"""Joint-space command shaper — the LAST line of defense before the motors.

Whatever the upstream pipeline commands (IK output, a bug, a teleport in the VR
stream), the joint command that leaves this object is guaranteed to be:

  - inside the PHYSICAL joint limits (the robot's hardstops, clamped here in
    software as well as on the Pinocchio model upstream),
  - slower than `rate_limit` rad/s per joint, ALWAYS — a target jump becomes a
    bounded-speed glide, never a snap,
  - smooth: a critically-damped second-order tracker (the command-side "PD")
    shapes accelerations, so the motor-side MIT PD receives a continuous,
    overshoot-free reference,
  - finite: a non-finite target holds the last safe command (fail-closed).

It also implements the soft-start that `safety.ramp_s` promised: on construction
(or `reset()`) the state initializes at the robot's CURRENT pose, so the first
engage glides from wherever the arm actually is.

This runs at the SINK boundary (hardware.py), deliberately decoupled from the
engine: simulation/render paths see the raw engine output, hardware sees the
shaped command, and the shaper is independently unit-tested.
"""
from __future__ import annotations

import numpy as np


class JointCommandShaper:
    def __init__(self, q0, *, rate_limit: float, smooth_hz: float, lo, hi):
        self.lo = np.asarray(lo, dtype=float).reshape(-1)
        self.hi = np.asarray(hi, dtype=float).reshape(-1)
        if not (np.all(np.isfinite(self.lo)) and np.all(np.isfinite(self.hi)) and np.all(self.lo < self.hi)):
            raise ValueError("shaper needs finite lo < hi joint limits")
        self.rate = float(rate_limit)
        if not (np.isfinite(self.rate) and self.rate > 0):
            raise ValueError("rate_limit must be finite and > 0")
        omega = 2.0 * np.pi * float(smooth_hz)
        if not (np.isfinite(omega) and omega > 0):
            raise ValueError("smooth_hz must be finite and > 0")
        self.kp = omega * omega          # critically damped: kp = ω², kd = 2ω
        self.kd = 2.0 * omega
        # Forward-Euler stability needs dt < 2/ω; clamp integration steps well under.
        self.max_dt = min(0.05, 0.5 / omega)
        self.q = np.clip(np.asarray(q0, dtype=float).reshape(-1), self.lo, self.hi)
        self.v = np.zeros_like(self.q)
        self._t: float | None = None

    def reset(self, q0, t: float | None = None) -> None:
        """Re-anchor at a (measured) pose with zero velocity — e.g. after an e-stop."""
        self.q = np.clip(np.asarray(q0, dtype=float).reshape(-1), self.lo, self.hi)
        self.v = np.zeros_like(self.q)
        self._t = t

    def shape(self, target, t: float) -> np.ndarray:
        """Advance the tracker toward `target` up to time `t`; returns the safe
        command. Long gaps are integrated in stable sub-steps so the rate limit
        holds across hiccups too."""
        tgt = np.asarray(target, dtype=float).reshape(-1)
        if tgt.shape != self.q.shape or not np.all(np.isfinite(tgt)):
            tgt = self.q.copy()                      # fail-closed: hold last safe
        tgt = np.clip(tgt, self.lo, self.hi)
        if self._t is None:
            self._t = float(t)
            return self.q.copy()
        remaining = max(0.0, min(float(t) - self._t, 0.5))   # cap runaway gaps
        self._t = float(t)
        while remaining > 1e-9:
            dt = min(remaining, self.max_dt)
            remaining -= dt
            a = self.kp * (tgt - self.q) - self.kd * self.v
            self.v = np.clip(self.v + a * dt, -self.rate, self.rate)
            self.q = np.clip(self.q + self.v * dt, self.lo, self.hi)
        return self.q.copy()
