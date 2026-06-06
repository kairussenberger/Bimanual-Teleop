"""One-Euro adaptive low-pass filtering — the canonical home (spec Section 9
`filters.py`, invariant #3: *all tracking input is filtered before it reaches IK*).

The One-Euro filter (Casiez, Roussel & Vogel, CHI 2012) smooths hard when the
signal is steady (kills jitter) and barely smooths during fast motion (kills lag),
by making the low-pass cutoff rise with the estimated speed. That speed/jitter
trade-off is exactly what teleop wants: a still wrist shouldn't shake the arm, a
fast wrist shouldn't lag it.

Two flavours, identical math:
  - `OneEuroFilter`     — a dict of independent scalar channels (used by the finger
                          retarget and the arm position target).
  - `OneEuroVecFilter`  — a fixed-length numpy vector, per-component adaptive cutoff
                          (convenient for filtering a position / quaternion / q-vector).

This module is the single implementation; `hands/retarget_core.OneEuroFilter` and
`arms/arm_control` import it. The defaults below are the proven finger-teleop
values ported from the webcam pipeline; the arm passes its own (snappier) params.
"""
from __future__ import annotations

import numpy as np

# Proven finger-teleop defaults (verbatim from the webcam pipeline). The arm path
# overrides these with snappier params (mincutoff≈4, beta≈1) for less lag.
DEFAULT_MINCUTOFF = 1.7
DEFAULT_BETA = 0.30
DEFAULT_DCUTOFF = 1.0

# Minimum dt (s) — guards a zero/negative/duplicate timestamp from blowing up the
# alpha computation (which divides by dt).
_MIN_DT = 1e-3


def _alpha(cutoff, dt: float):
    """Low-pass smoothing factor for a given cutoff (Hz) and timestep (s). Works for
    a scalar OR an elementwise numpy array of cutoffs."""
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One-Euro filter over a dict of independent scalar channels. One instance per
    stream (e.g. per hand); state persists across calls. Math is verbatim from the
    proven webcam_teleop implementation."""

    def __init__(self, mincutoff: float = DEFAULT_MINCUTOFF,
                 beta: float = DEFAULT_BETA, dcutoff: float = DEFAULT_DCUTOFF):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self._x_prev: dict = {}
        self._dx_prev: dict = {}
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        return float(_alpha(cutoff, dt))

    def __call__(self, values: dict, t: float) -> dict:
        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = dict(values)
            self._dx_prev = {k: 0.0 for k in values}
            return dict(values)
        dt = max(t - self._t_prev, _MIN_DT)
        self._t_prev = t
        out = {}
        for k, x in values.items():
            x_prev = self._x_prev.get(k, x)
            dx = (x - x_prev) / dt
            a_d = self._alpha(self.dcutoff, dt)
            dx_hat = a_d * dx + (1 - a_d) * self._dx_prev.get(k, 0.0)
            cutoff = self.mincutoff + self.beta * abs(dx_hat)
            a = self._alpha(cutoff, dt)
            x_hat = a * x + (1 - a) * x_prev
            self._x_prev[k], self._dx_prev[k] = x_hat, dx_hat
            out[k] = x_hat
        return out


class OneEuroVecFilter:
    """One-Euro filter over a fixed-length numpy vector, with a per-component
    adaptive cutoff. Convenient for a 3-vector position or a joint-target vector."""

    def __init__(self, mincutoff: float = DEFAULT_MINCUTOFF,
                 beta: float = DEFAULT_BETA, dcutoff: float = DEFAULT_DCUTOFF):
        self.mincutoff, self.beta, self.dcutoff = mincutoff, beta, dcutoff
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x_prev = self._dx_prev = self._t_prev = None

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            return x.copy()
        dt = max(t - self._t_prev, _MIN_DT)
        self._t_prev = t
        dx = (x - self._x_prev) / dt
        a_d = _alpha(self.dcutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.mincutoff + self.beta * np.abs(dx_hat)
        a = _alpha(cutoff, dt)               # elementwise
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev, self._dx_prev = x_hat, dx_hat
        return x_hat
