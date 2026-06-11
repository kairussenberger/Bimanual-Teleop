"""Operator-triggered TWO-POSE calibration (position-only, runtime).

The absolute mapping is 1:1 by default: your torso→wrist vector in metres becomes
the robot's chest→wrist vector. Two things break a naive fit of that map:

  1. Operators are not robot-sized (the YAM's reach/mounting proportions differ).
  2. ORBIT's hand positions live in a RECENTER-ANCHORED frame: starting the app
     or recentering with the headset somewhere arbitrary (a desk) shifts EVERY
     hand position by an unknown constant 3-vector (measured: 0.5 m down after a
     desk start — a one-pose fit then reads "arms at hip height" and either
     refuses or fits garbage). There is no in-data absolute reference: the hand
     keypoints share the same anchor as the wrist stream.

The guided THREE-POSE capture solves both at once:

    pose A — extend both arms straight forward at shoulder height, hold ~2.5 s
    pose B — relax both arms down at your sides, hold ~2.5 s
    pose C — press your palms together in front of your chest, hold ~2.5 s
             (anchors the LATERAL map at contact width: YOUR clap maps to the
             ROBOT's hands touching, by construction, and your true midline
             is measured where your palms meet)

Everything the fit needs comes out anchor-proof and head-yaw-proof:
  - the operator's FORWARD direction = the horizontal direction of the A−B
    wrist-midpoint delta (raising your arms from your sides to extended-forward
    IS forward — wherever your head points, e.g. at the dashboard);
  - LATERAL scale from the pose-A wrist SPREAD (anchor cancels in the spread);
  - FORWARD and UP scales from the per-axis A−B DELTAS against the robot's
    matching references (robot_neutral_wrist ↔ A, robot_rest_wrist — its actual
    rest pose — ↔ B): the anchor cancels in every difference;
  - the OFFSET (computed last, from pose A) absorbs whatever the anchor did;
    the operator's measured midline (`lat_center`) maps to the robot's midline.

ORIENTATION IS NEVER TOUCHED. The absolute attitude mapping stays
calibration-free by contract (see CLAUDE.md / tests/test_frames.py).

The result is applied inside ClutchMapper._p_abs and persisted as JSON
(per-machine, gitignored). NOTE: recentering the headset mid-session moves the
anchor again — if the mapping suddenly feels shifted, recalibrate (8 seconds)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import SIDES

# Capture gates.
HOLD_S = 2.5          # continuous still time to accept each pose
WINDOW_S = 0.6        # rolling stillness window
STILL_TOL = 0.030     # m — std-norm of wrist_body over the window
SPREAD_MIN = 0.20     # m — pose-A lateral wrist spread (right − left), anchor-proof
SPREAD_MAX = 0.80
DROP_MIN = 0.22       # m — pose-B wrists must sit at least this far BELOW pose A
DELTA_MIN = 0.15      # m — horizontal A−B midpoint delta needed to define forward
CLAP_SPREAD_MAX = 0.18  # m — pose-C wrists must be close together (palms pressed)
RAISE_MIN = 0.15      # m — pose-C wrists must come back UP from pose B
TIMEOUT_S = 180.0     # both poses
SCALE_MIN, SCALE_MAX = 0.6, 2.0
OFFSET_MAX = 0.80     # m — must cover anatomy AND the recenter-anchor shift

# Robot-side references (body coords [right, up, forward] relative to the chest
# anchor) used when the rig does not provide them. Probed with the real IK.
ROBOT_NEUTRAL_DEFAULT = {"left": (-0.22, 0.02, 0.46), "right": (0.22, 0.02, 0.46)}
ROBOT_REST_DEFAULT = {"left": (-0.221, -0.437, -0.032), "right": (0.222, -0.440, 0.032)}


@dataclass
class CalibResult:
    """A fitted position calibration, in OPERATOR body axes:
    out_lat = s_lat-ramp(lat − lat_center); out_up/fwd = S·in + offset.
    `lat_ref` = half the measured pose-A spread (the non-linear lateral ramp
    reaches full scale there); `lat_center` = the operator's measured midline
    (absorbs the anchor's lateral shift — maps to the robot's midline)."""
    axis_scale: np.ndarray                  # (3,) [right, up, forward]
    body_offset: np.ndarray                 # (3,) metres ([0] unused — see lat_center)
    lat_ref: float = 0.0
    lat_center: float = 0.0
    # Piecewise-linear lateral curve knots [[x_clap, y_contact], [x_spread,
    # y_robot_half_spread]] (|lat−center| → robot |lat|) from pose C; None →
    # the legacy quadratic ramp. forward_body = the operator's arm-defined
    # forward (2D, measured body frame) — the engine latches its yaw lock to it.
    lat_knots: list | None = None
    forward_body: list | None = None
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"axis_scale": [round(float(v), 3) for v in self.axis_scale],
                "body_offset": [round(float(v), 3) for v in self.body_offset],
                "lat_ref": round(float(self.lat_ref), 3),
                "lat_center": round(float(self.lat_center), 3),
                "lat_knots": ([[round(float(v), 3) for v in k] for k in self.lat_knots]
                              if self.lat_knots else None),
                "stamp": self.meta.get("stamp")}

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 4,
                   "axis_scale": [float(v) for v in self.axis_scale],
                   "body_offset": [float(v) for v in self.body_offset],
                   "lat_ref": float(self.lat_ref),
                   "lat_center": float(self.lat_center),
                   "lat_knots": ([[float(v) for v in k] for k in self.lat_knots]
                                 if self.lat_knots else None),
                   "forward_body": ([float(v) for v in self.forward_body]
                                    if self.forward_body else None),
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
    lat_center = float(d.get("lat_center", 0.0))
    meta = d.get("meta", {})
    if lat_ref <= 0.0 and isinstance(meta.get("op_neutral"), dict):
        try:   # version-1/2 files: derive the ramp reference from the stored neutral
            lat_ref = float(np.mean([abs(meta["op_neutral"][s][0]) for s in SIDES]))
        except (KeyError, TypeError, IndexError):
            lat_ref = 0.0
    if not (0.0 <= lat_ref <= 0.6) or not np.isfinite(lat_center) or abs(lat_center) > 0.6:
        lat_ref, lat_center = max(0.0, min(lat_ref, 0.6)) if np.isfinite(lat_ref) else 0.0, 0.0
    knots = d.get("lat_knots")
    if knots is not None:
        try:
            knots = [[float(a), float(b)] for a, b in knots]
            ok = (len(knots) == 2 and 0.0 < knots[0][0] < knots[1][0] <= 0.6
                  and 0.0 < knots[0][1] < knots[1][1] <= 0.6)
            knots = knots if ok else None
        except (TypeError, ValueError):
            knots = None
    fb = d.get("forward_body")
    return CalibResult(axis_scale=scale, body_offset=off, lat_ref=lat_ref,
                       lat_center=lat_center, lat_knots=knots,
                       forward_body=list(fb) if fb else None, meta=meta)


def _reyaw_frame(mid_a: np.ndarray, mid_b: np.ndarray):
    """Forward/right horizontal unit vectors from the A−B midpoint delta
    (raising the arms from the sides to extended-forward IS forward). Returns
    (f2, r2) 2-vectors over (lat, fwd) components, or None if the delta is too
    small to define a direction."""
    d = np.asarray(mid_a, float) - np.asarray(mid_b, float)
    h = np.array([d[0], d[2]])
    n = float(np.linalg.norm(h))
    if n < DELTA_MIN:
        return None
    f2 = h / n
    r2 = np.array([f2[1], -f2[0]])
    return f2, r2


def _reyaw(v: np.ndarray, f2: np.ndarray, r2: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    xz = np.array([v[0], v[2]])
    return np.array([float(xz @ r2), v[1], float(xz @ f2)])


def fit_two_pose(pose_a: dict[str, np.ndarray], pose_b: dict[str, np.ndarray],
                 robot_neutral: dict[str, np.ndarray], robot_rest: dict[str, np.ndarray],
                 pos_scale: float = 1.0, pose_c: dict[str, np.ndarray] | None = None,
                 robot_clap_gap: float = 0.12) -> CalibResult | None:
    """Fit scale/offset from the held poses. Every scale comes from an A−B
    DIFFERENCE or a spread, so the ORBIT recenter anchor cancels exactly; the
    offset (from pose A) absorbs it. With pose C (palms together) the lateral
    map becomes a piecewise-linear curve anchored at CONTACT width — the
    operator's clap maps to the robot's hands touching by construction — and
    the midline is measured where the palms actually meet. Pure math."""
    mid_a = 0.5 * (np.asarray(pose_a["left"], float) + np.asarray(pose_a["right"], float))
    mid_b = 0.5 * (np.asarray(pose_b["left"], float) + np.asarray(pose_b["right"], float))
    fr = _reyaw_frame(mid_a, mid_b)
    if fr is None:
        return None
    f2, r2 = fr
    A = {s: _reyaw(pose_a[s], f2, r2) for s in SIDES}
    B = {s: _reyaw(pose_b[s], f2, r2) for s in SIDES}
    rbN = {s: np.asarray(robot_neutral[s], dtype=float).reshape(3) for s in SIDES}
    rbR = {s: np.asarray(robot_rest[s], dtype=float).reshape(3) for s in SIDES}

    spread_a = A["right"][0] - A["left"][0]
    if not (SPREAD_MIN <= spread_a <= SPREAD_MAX):
        return None
    s_lat = (rbN["right"][0] - rbN["left"][0]) / spread_a
    d_up = float(np.mean([A[s][1] - B[s][1] for s in SIDES]))
    d_fwd = float(np.mean([A[s][2] - B[s][2] for s in SIDES]))
    if d_up < DROP_MIN or d_fwd < DELTA_MIN / 2:
        return None
    s_up = float(np.mean([rbN[s][1] - rbR[s][1] for s in SIDES])) / d_up
    s_fwd = float(np.mean([rbN[s][2] - rbR[s][2] for s in SIDES])) / d_fwd
    scale = np.clip(np.array([s_lat, s_up, s_fwd]), SCALE_MIN, SCALE_MAX)

    lat_center = 0.5 * (A["right"][0] + A["left"][0])    # operator midline (incl. anchor)
    lat_knots = None
    C = None
    if pose_c is not None:
        C = {s: _reyaw(pose_c[s], f2, r2) for s in SIDES}
        clap_gap = C["right"][0] - C["left"][0]
        if -0.05 <= clap_gap <= CLAP_SPREAD_MAX:
            lat_center = 0.5 * (C["right"][0] + C["left"][0])   # palms meet AT the midline
            x_c = max(abs(clap_gap) / 2.0, 0.01)
            y_c = max(robot_clap_gap, 0.02) / 2.0
            x_a = spread_a / 2.0
            y_a = 0.5 * (rbN["right"][0] - rbN["left"][0])
            if x_c < x_a - 0.02 and y_c < y_a:
                lat_knots = [[x_c, y_c], [x_a, y_a]]
    ps = max(pos_scale, 1e-6)
    off = np.mean([rbN[s] / ps - scale * A[s] for s in SIDES], axis=0)
    off = np.clip(off, -OFFSET_MAX, OFFSET_MAX)
    off[0] = 0.0                                          # lateral handled by lat_center
    meta = {"stamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pose_a": {s: [round(float(v), 4) for v in A[s]] for s in SIDES},
            "pose_b": {s: [round(float(v), 4) for v in B[s]] for s in SIDES},
            "pose_c": ({s: [round(float(v), 4) for v in C[s]] for s in SIDES}
                       if C is not None else None),
            "op_neutral": {s: [round(float(v), 4) for v in A[s]] for s in SIDES},
            "robot_neutral": {s: rbN[s].round(4).tolist() for s in SIDES},
            "robot_rest": {s: rbR[s].round(4).tolist() for s in SIDES}}
    return CalibResult(axis_scale=scale, body_offset=off, lat_ref=spread_a / 2.0,
                       lat_center=lat_center, lat_knots=lat_knots,
                       forward_body=[float(f2[0]), float(f2[1])], meta=meta)


class NeutralPoseCalibration:
    """The guided two-pose capture: A (arms extended forward) then B (arms
    relaxed at the sides), each held still HOLD_S. Drives `status` dicts the
    dashboard renders as the prompt. Clock-injected and deterministic; the
    engine feeds it body-relative wrist samples each tick."""

    def __init__(self, rig: dict):
        m = rig.get("mapping", {})
        rn = m.get("robot_neutral_wrist") or {}
        rr = m.get("robot_rest_wrist") or {}
        self.robot_neutral = {
            s: np.asarray(rn.get(s, ROBOT_NEUTRAL_DEFAULT[s]), dtype=float).reshape(3)
            for s in SIDES}
        self.robot_rest = {
            s: np.asarray(rr.get(s, ROBOT_REST_DEFAULT[s]), dtype=float).reshape(3)
            for s in SIDES}
        self.pos_scale = float(m.get("pos_scale", 1.0))
        self.robot_clap_gap = float(m.get("robot_clap_gap", 0.12))
        self.active = False
        self.phase = "idle"      # idle | wait_fwd | wait_rest | wait_clap | done | cancelled
        self.result: CalibResult | None = None
        self._t0 = 0.0
        self._hold_t0: float | None = None
        self._buf: dict[str, list[tuple[float, np.ndarray]]] = {s: [] for s in SIDES}
        self._pose_a: dict[str, np.ndarray] | None = None
        self._pose_b: dict[str, np.ndarray] | None = None
        self._msg = ""
        self._seen = {s: False for s in SIDES}

    # ---- lifecycle --------------------------------------------------------- #
    def start(self, t: float) -> None:
        self.active = True
        self.phase = "wait_fwd"
        self.result = None
        self._t0 = t
        self._hold_t0 = None
        self._buf = {s: [] for s in SIDES}
        self._pose_a = None
        self._pose_b = None
        self._msg = ""

    def cancel(self, msg: str = "calibration cancelled") -> None:
        self.active = False
        self.phase = "cancelled"
        self._msg = msg

    # ---- per-tick ---------------------------------------------------------- #
    def tick(self, wrist_body: dict[str, np.ndarray | None], t: float) -> None:
        if not self.active:
            return
        if (t - self._t0) > TIMEOUT_S:
            self.cancel("calibration timed out — press CALIBRATE to retry")
            return
        for s in SIDES:
            w = wrist_body.get(s)
            self._seen[s] = w is not None
            if w is not None:
                buf = self._buf[s]
                buf.append((t, np.asarray(w, dtype=float).reshape(3)))
                while buf and (t - buf[0][0]) > max(WINDOW_S, HOLD_S):
                    buf.pop(0)
        ready = self._pose_ready(t)
        if ready:
            if self._hold_t0 is None:
                self._hold_t0 = t
            elif (t - self._hold_t0) >= HOLD_S:
                self._advance(t)
        else:
            self._hold_t0 = None

    def _window(self, side: str, t: float, span: float) -> np.ndarray | None:
        pts = [w for (ts, w) in self._buf[side] if (t - ts) <= span]
        return np.stack(pts) if len(pts) >= 4 else None

    def _still_means(self, t: float) -> dict[str, np.ndarray] | None:
        """Per-side window means, or None unless BOTH hands are tracked + still."""
        means = {}
        for s in SIDES:
            win = self._window(s, t, WINDOW_S)
            if win is None or float(np.linalg.norm(win.std(axis=0))) > STILL_TOL:
                return None
            means[s] = win.mean(axis=0)
        return means

    def _pose_ready(self, t: float) -> bool:
        """Anchor-proof gates (nothing trusts an absolute frame): pose A needs a
        sane lateral SPREAD; pose B needs the wrists DROPPED ≥ DROP_MIN below
        pose A; pose C (palms together) needs the wrists CLOSE and raised back
        up from pose B."""
        means = self._still_means(t)
        if means is None:
            return False
        spread = abs(means["right"][0] - means["left"][0])
        if self.phase == "wait_clap":
            raised = float(np.mean([means[s][1] - self._pose_b[s][1] for s in SIDES]))
            return spread <= CLAP_SPREAD_MAX and raised >= RAISE_MIN
        if not (SPREAD_MIN <= spread <= SPREAD_MAX):
            return False
        if self.phase in ("wait_fwd", "hold"):
            return True
        drop = float(np.mean([self._pose_a[s][1] - means[s][1] for s in SIDES]))
        return drop >= DROP_MIN

    def _advance(self, t: float) -> None:
        means = {}
        for s in SIDES:
            win = self._window(s, t, HOLD_S)
            if win is None:
                self._hold_t0 = None
                return
            means[s] = win.mean(axis=0)
        if self.phase in ("wait_fwd", "hold"):
            self._pose_a = means
            self.phase = "wait_rest"
            self._hold_t0 = None
            self._buf = {s: [] for s in SIDES}       # fresh windows for the next pose
            return
        if self.phase == "wait_rest":
            self._pose_b = means
            self.phase = "wait_clap"
            self._hold_t0 = None
            self._buf = {s: [] for s in SIDES}
            return
        res = fit_two_pose(self._pose_a, self._pose_b, self.robot_neutral, self.robot_rest,
                           self.pos_scale, pose_c=means, robot_clap_gap=self.robot_clap_gap)
        if res is None:                               # degenerate capture — keep waiting
            self._hold_t0 = None
            return
        self.result = res
        self.active = False
        self.phase = "done"
        sc = res.axis_scale
        self._msg = (f"CALIBRATED ✓ scale lat {sc[0]:.2f} / up {sc[1]:.2f} / reach {sc[2]:.2f}, "
                     f"midline {res.lat_center:+.2f} m"
                     + ("" if res.lat_knots else " (no clap anchor)"))

    # ---- display ----------------------------------------------------------- #
    def status(self, t: float) -> dict:
        if self.active and self._hold_t0 is not None:
            elapsed = t - self._hold_t0
            step = {"wait_fwd": "1/3", "hold": "1/3", "wait_rest": "2/3"}.get(self.phase, "3/3")
            return {"active": True, "kind": "neutral", "phase": "hold",
                    "progress": min(1.0, elapsed / HOLD_S),
                    "remaining": max(0.0, HOLD_S - elapsed),
                    "left": self._seen["left"], "right": self._seen["right"],
                    "msg": f"HOLD STILL ({step}) — measuring… {max(0.0, HOLD_S - elapsed):.1f}s"}
        if self.active:
            if not all(self._seen.values()):
                msg = "CALIBRATION: wear the headset, controllers down — both hands in view"
            elif self.phase == "wait_rest":
                msg = "CALIBRATION 2/3: now RELAX BOTH ARMS DOWN at your sides — and hold"
            elif self.phase == "wait_clap":
                msg = "CALIBRATION 3/3: PRESS YOUR PALMS TOGETHER in front of your chest — and hold"
            else:
                msg = "CALIBRATION 1/3: EXTEND BOTH ARMS straight forward at shoulder height"
            return {"active": True, "kind": "neutral", "phase": self.phase, "progress": 0.0,
                    "remaining": HOLD_S, "left": self._seen["left"],
                    "right": self._seen["right"], "msg": msg}
        return {"active": False, "kind": "neutral", "phase": self.phase,
                "progress": 1.0 if self.phase == "done" else 0.0, "remaining": 0.0,
                "left": self._seen["left"], "right": self._seen["right"], "msg": self._msg}
