"""Operator-triggered NEUTRAL-POSE calibration (position-only, runtime).

The absolute mapping is 1:1 by default: your torso→wrist vector in metres becomes
the robot's chest→wrist vector. Operators are not robot-sized — the YAM's reach
and mounting proportions differ from a human arm — so a pure 1:1 map either
wastes robot workspace or parks the robot's neutral somewhere that feels wrong.
This module fits the POSITION mapping to the operator from one guided pose:

    extend both arms straight forward at shoulder height, hold still ~2.5 s

From the per-side mean torso→wrist vector (body axes [right, up, forward]) it
derives:
  - a LATERAL scale   s_lat = |robot neutral lateral| / |operator neutral lateral|
    (operator hand spacing → robot hand spacing; the midline stays the midline,
    so bringing your hands together still brings the robot's hands together),
  - a REACH scale     s_fwd = robot neutral forward / operator neutral forward,
    shared by the vertical axis (both are arm-length bound),
  - an UP/FORWARD offset aligning your neutral with the robot's neutral
    (lateral offset forced to 0 — never bias one side across the midline).

ORIENTATION IS NEVER TOUCHED. The absolute attitude mapping stays
calibration-free by contract: a stance-calibrated orientation correspondence
measured ~145° median axis error on a real session and must not return
(see CLAUDE.md / tests/test_frames.py).

The result is applied inside ClutchMapper._p_abs (body axes, before the
body→base rotation) and persisted as JSON (per-machine, gitignored)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import SIDES

# Capture gates. The pose gate rejects arms-at-sides / T-pose holds so a
# distracted "calibration" cannot bake junk scales in.
HOLD_S = 2.5          # continuous still time required to accept the pose
WINDOW_S = 0.6        # rolling stillness window
STILL_TOL = 0.030     # m — std-norm of wrist_body over the window (allows breathing sway)
MIN_FORWARD = 0.25    # m — wrists must be extended forward at least this far
MIN_LATERAL = 0.03    # m — and not crossed over the body midline
TIMEOUT_S = 120.0     # give up (un-freeze the arms) if never completed
SCALE_MIN, SCALE_MAX = 0.6, 2.0
OFFSET_MAX = 0.40     # m — per-axis cap on the fitted offset

# Robot-side neutral reference (body coords [right, up, forward], m, relative to
# the chest anchor) used when the rig does not provide mapping.robot_neutral_wrist.
# Values probed with the real IK (see config/rig.yaml comment).
ROBOT_NEUTRAL_DEFAULT = {"left": (-0.22, 0.02, 0.46), "right": (0.22, 0.02, 0.46)}


@dataclass
class CalibResult:
    """A fitted position calibration: body-axes per-axis scale + offset.
    `lat_ref` = the operator's neutral lateral |y| — the lateral scale ramps
    from 1:1 at the midline (claps stay claps) to axis_scale[0] at lat_ref."""
    axis_scale: np.ndarray                  # (3,) [right, up, forward]
    body_offset: np.ndarray                 # (3,) metres, body axes
    lat_ref: float = 0.0                    # m; 0 = legacy linear lateral scale
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """JSON-safe summary for the render stream / dashboard chip."""
        return {"axis_scale": [round(float(v), 3) for v in self.axis_scale],
                "body_offset": [round(float(v), 3) for v in self.body_offset],
                "lat_ref": round(float(self.lat_ref), 3),
                "stamp": self.meta.get("stamp")}

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 2,
                   "axis_scale": [float(v) for v in self.axis_scale],
                   "body_offset": [float(v) for v in self.body_offset],
                   "lat_ref": float(self.lat_ref),
                   "meta": self.meta}
        p.write_text(json.dumps(payload, indent=2) + "\n")


def load_calibration(path: str | Path) -> CalibResult | None:
    """Load + validate a persisted calibration; None when absent or implausible
    (a corrupt/out-of-range file must never steer the arms)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
        scale = np.asarray(d["axis_scale"], dtype=float).reshape(3)
        off = np.asarray(d["body_offset"], dtype=float).reshape(3)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
        return None
    if not (np.all(np.isfinite(scale)) and np.all(np.isfinite(off))):
        return None
    if np.any(scale < SCALE_MIN - 1e-9) or np.any(scale > SCALE_MAX + 1e-9):
        return None
    if np.any(np.abs(off) > OFFSET_MAX + 1e-9):
        return None
    lat_ref = float(d.get("lat_ref", 0.0))
    meta = d.get("meta", {})
    if lat_ref <= 0.0 and isinstance(meta.get("op_neutral"), dict):
        # version-1 file: derive the ramp reference from the stored neutral
        try:
            lat_ref = float(np.mean([abs(meta["op_neutral"][s][0]) for s in SIDES]))
        except (KeyError, TypeError, IndexError):
            lat_ref = 0.0
    if not (0.0 <= lat_ref <= 0.6):
        lat_ref = 0.0
    return CalibResult(axis_scale=scale, body_offset=off, lat_ref=lat_ref, meta=meta)


def fit_neutral(op_neutral: dict[str, np.ndarray], robot_neutral: dict[str, np.ndarray],
                pos_scale: float = 1.0) -> CalibResult:
    """Fit scale/offset so the operator's measured neutral lands on the robot's
    neutral: chest + pos_scale·R·(offset + S ⊙ wrist_body) per side.

    Pure math, no state — unit-testable. Scales are clamped to sane ranges and
    the lateral offset is forced to zero (midline stays midline)."""
    s_lat, s_fwd = [], []
    for s in SIDES:
        op = np.asarray(op_neutral[s], dtype=float).reshape(3)
        rb = np.asarray(robot_neutral[s], dtype=float).reshape(3)
        s_lat.append(abs(rb[0]) / max(abs(op[0]), MIN_LATERAL))
        s_fwd.append(max(rb[2], 0.05) / max(op[2], 0.10))
    lat = float(np.clip(np.mean(s_lat), SCALE_MIN, SCALE_MAX))
    fwd = float(np.clip(np.mean(s_fwd), SCALE_MIN, SCALE_MAX))
    scale = np.array([lat, fwd, fwd])       # vertical shares the reach scale
    offs = []
    for s in SIDES:
        op = np.asarray(op_neutral[s], dtype=float).reshape(3)
        rb = np.asarray(robot_neutral[s], dtype=float).reshape(3)
        offs.append(rb / max(pos_scale, 1e-6) - scale * op)
    off = np.clip(np.mean(offs, axis=0), -OFFSET_MAX, OFFSET_MAX)
    off[0] = 0.0
    meta = {"stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "op_neutral": {s: [round(float(v), 4) for v in np.asarray(op_neutral[s]).reshape(3)]
                           for s in SIDES},
            "robot_neutral": {s: [round(float(v), 4) for v in np.asarray(robot_neutral[s]).reshape(3)]
                              for s in SIDES}}
    lat_ref = float(np.mean([abs(np.asarray(op_neutral[s], dtype=float)[0]) for s in SIDES]))
    return CalibResult(axis_scale=scale, body_offset=off, lat_ref=lat_ref, meta=meta)


class NeutralPoseCalibration:
    """The guided capture: wait for both hands extended-forward and still, hold
    HOLD_S, then fit. Drives `status` dicts the dashboard renders as the prompt.

    Clock-injected (every entry point takes `t`) so it is deterministic and
    testable; the engine feeds it body-relative wrist samples each tick."""

    def __init__(self, rig: dict):
        m = rig.get("mapping", {})
        rn = m.get("robot_neutral_wrist") or {}
        self.robot_neutral = {
            s: np.asarray(rn.get(s, ROBOT_NEUTRAL_DEFAULT[s]), dtype=float).reshape(3)
            for s in SIDES}
        self.pos_scale = float(m.get("pos_scale", 1.0))
        self.active = False
        self.phase = "idle"                 # idle | wait | hold | done | cancelled
        self.result: CalibResult | None = None
        self._t0 = 0.0
        self._hold_t0: float | None = None
        self._buf: dict[str, list[tuple[float, np.ndarray]]] = {s: [] for s in SIDES}
        self._msg = ""
        self._seen = {s: False for s in SIDES}

    # ---- lifecycle --------------------------------------------------------- #
    def start(self, t: float) -> None:
        self.active = True
        self.phase = "wait"
        self.result = None
        self._t0 = t
        self._hold_t0 = None
        self._buf = {s: [] for s in SIDES}
        self._msg = ""

    def cancel(self, msg: str = "calibration cancelled") -> None:
        self.active = False
        self.phase = "cancelled"
        self._msg = msg

    # ---- per-tick ---------------------------------------------------------- #
    def tick(self, wrist_body: dict[str, np.ndarray | None], t: float) -> None:
        """Feed the current per-side body-relative wrist positions (None = not
        tracked). Advances wait→hold→done."""
        if not self.active:
            return
        if (t - self._t0) > TIMEOUT_S:
            self.cancel("calibration timed out — press CALIBRATE to retry")
            return
        ready = {}
        for s in SIDES:
            w = wrist_body.get(s)
            self._seen[s] = w is not None
            if w is not None:
                buf = self._buf[s]
                buf.append((t, np.asarray(w, dtype=float).reshape(3)))
                while buf and (t - buf[0][0]) > max(WINDOW_S, HOLD_S):
                    buf.pop(0)
            ready[s] = self._side_ready(s, t)
        if all(ready.values()):
            if self._hold_t0 is None:
                self._hold_t0 = t
            elif (t - self._hold_t0) >= HOLD_S:
                self._finish(t)
                return
        else:
            self._hold_t0 = None

    def _window(self, side: str, t: float, span: float) -> np.ndarray | None:
        pts = [w for (ts, w) in self._buf[side] if (t - ts) <= span]
        return np.stack(pts) if len(pts) >= 4 else None

    def _side_ready(self, side: str, t: float) -> bool:
        """Tracked + still + plausibly in the extended-forward pose."""
        win = self._window(side, t, WINDOW_S)
        if win is None:
            return False
        if float(np.linalg.norm(win.std(axis=0))) > STILL_TOL:
            return False
        mean = win.mean(axis=0)
        expected = -1.0 if side == "left" else 1.0
        if mean[0] * expected < MIN_LATERAL:        # crossed over the midline
            return False
        return mean[2] >= MIN_FORWARD               # actually extended forward

    def _finish(self, t: float) -> None:
        means = {}
        for s in SIDES:
            win = self._window(s, t, HOLD_S)
            if win is None:                          # lost samples at the last instant
                self._hold_t0 = None
                return
            means[s] = win.mean(axis=0)
        self.result = fit_neutral(means, self.robot_neutral, self.pos_scale)
        self.active = False
        self.phase = "done"
        sc, of = self.result.axis_scale, self.result.body_offset
        self._msg = (f"CALIBRATED ✓ scale lat {sc[0]:.2f} / reach {sc[2]:.2f}, "
                     f"offset up {of[1]:+.2f} fwd {of[2]:+.2f} m")

    # ---- display ----------------------------------------------------------- #
    def status(self, t: float) -> dict:
        """Same shape as the legacy stillness-hold status (the dashboard banner
        and the in-headset HUD already render this dict)."""
        if self.active and self._hold_t0 is not None:
            elapsed = t - self._hold_t0
            return {"active": True, "kind": "neutral", "phase": "hold",
                    "progress": min(1.0, elapsed / HOLD_S),
                    "remaining": max(0.0, HOLD_S - elapsed),
                    "left": self._seen["left"], "right": self._seen["right"],
                    "msg": f"HOLD STILL — measuring… {max(0.0, HOLD_S - elapsed):.1f}s"}
        if self.active:
            missing = [s for s in SIDES if not self._side_ready(s, t)]
            if not all(self._seen.values()):
                msg = "CALIBRATION: wear the headset, controllers down — both hands in view"
            else:
                msg = ("CALIBRATION: EXTEND BOTH ARMS straight forward at shoulder height"
                       + (f" — adjust {' & '.join(missing).upper()}" if missing else ""))
            return {"active": True, "kind": "neutral", "phase": "wait", "progress": 0.0,
                    "remaining": HOLD_S, "left": self._seen["left"],
                    "right": self._seen["right"], "msg": msg}
        return {"active": False, "kind": "neutral", "phase": self.phase,
                "progress": 1.0 if self.phase == "done" else 0.0, "remaining": 0.0,
                "left": self._seen["left"], "right": self._seen["right"], "msg": self._msg}
