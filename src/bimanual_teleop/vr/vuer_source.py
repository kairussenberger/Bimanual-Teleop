"""Real Quest 3 / 3S hand-tracking over WebXR, via Vuer (OpenTeleVision approach).

The operator opens the Vuer page in the Quest browser and enters immersive mode;
the WebXR runtime streams both wrists' 6-DoF poses and 25 joints/hand back over a
websocket. We mirror the latest into a thread-safe VRFrame. Sideload-free and
macOS-native (no OpenXR runtime). WebXR requires HTTPS — provide a cert (mkcert)
or set vr.ngrok: true.

Install: `uv sync --extra vr`. Cert: `mkcert -install && mkcert <PC-LAN-IP>` →
point vr.cert_file/key_file at the result (or use ngrok).

Vuer event payload (per docs/OpenTeleVision): HAND_MOVE event.value has
leftHand/rightHand = 16 floats (column-major 4x4 wrist pose) and
leftLandmarks/rightLandmarks = 75 floats (25 joints × xyz, W3C order). CAMERA_MOVE
carries the head pose. Keys can vary across Vuer/browser versions — the parsing is
defensive; verify against your version with --debug.
"""
from __future__ import annotations

import dataclasses
import threading
import time

import numpy as np

from ..config import SIDES
from .frames import HandSample, VRFrame
from .ingest import VRSource


def _mat4(flat) -> np.ndarray:
    a = np.asarray(flat, dtype=float)
    if a.size != 16:
        return np.eye(4)
    return a.reshape(4, 4, order="F")   # WebXR/Vuer matrices are column-major


def _pinch_from_landmarks(lm) -> float:
    """Thumb-tip↔index-tip distance normalized by hand size → 1 pinched, 0 open."""
    if lm is None or len(lm) < 25:
        return 0.0
    d = np.linalg.norm(lm[4] - lm[9])            # thumb tip (4) ↔ index tip (9)
    scale = np.linalg.norm(lm[11] - lm[0]) + 1e-6  # wrist → middle proximal
    return float(np.clip((0.6 - d / scale) / (0.6 - 0.2), 0.0, 1.0))


# Fixed floating-panel positions in the WebXR floor frame (+Y up, −Z forward):
# a triad for each hand at face height, ~0.6 m in front. YOUR frame on top, the
# ROBOT's frame just below it, so you compare orientation per side. The hand poses
# are wrist-RELATIVE (wrist position reads ~0 = on the floor), so pinning the viz
# at a fixed spot is the only way it stays visible once you enter immersive VR.
_VIZ_POS = {
    "op_left":  [-0.30, 1.45, -0.6], "rb_left":  [-0.30, 1.15, -0.6],
    "op_right": [0.30, 1.45, -0.6],  "rb_right": [0.30, 1.15, -0.6],
}


def _coords_mat(pos, R) -> list:
    """Col-major 4x4 (Vuer/WebXR convention) from a position + 3x3 rotation."""
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = pos
    return list(M.reshape(-1, order="F"))


class VuerVRSource(VRSource):
    def __init__(self, rig: dict, debug: bool = False):
        v = rig.get("vr", {})
        self.cert = v.get("cert_file", "cert.pem")
        self.key = v.get("key_file", "key.pem")
        # tunnel mode: serve plain HTTP on localhost; a cloudflared/ngrok tunnel
        # provides the public HTTPS the Quest connects to (works on isolated
        # campus Wi-Fi, no self-signed cert warning). Else: HTTPS on the LAN.
        self.tunnel = bool(v.get("tunnel", False))
        self.debug = bool(v.get("debug", debug))
        self._dbg = {"hand": 0, "cam": 0, "ctrl": 0, "last": ""}
        self._lock = threading.Lock()
        self._frame = VRFrame(hands={s: HandSample() for s in SIDES})
        self._robot_viz: dict[str, list] = {}   # side -> 16-float col-major matrix (robot hand frame, WebXR)
        self._calib: dict | None = None          # in-headset calibration banner state
        self._hud: list[str] = []                # in-headset live status/log lines
        self._thread: threading.Thread | None = None
        self._app = None

    def set_robot_frame(self, side: str, R3x3) -> None:
        """Push the robot's hand ORIENTATION (3x3, in the WebXR/headset frame) for
        the in-headset viz. Position is fixed (floating panel), so only orientation
        matters for the comparison. Called from the control loop."""
        with self._lock:
            self._robot_viz[side] = np.asarray(R3x3, dtype=float).reshape(3, 3)

    def set_calib(self, status) -> None:
        """Push the calibration banner state for the in-headset countdown (engine
        publishes it each tick as engine.calib_status). None = hide the banner."""
        with self._lock:
            self._calib = dict(status) if status else None

    def set_hud(self, lines) -> None:
        """Push a list of status/log text lines to render in-headset (called every
        control tick). This is the in-headset 'logs' panel: tracking flags, wrist
        roll, clutch/calib state, loop rate — so the operator is never flying blind.
        Pass [] / None to clear it."""
        with self._lock:
            self._hud = [str(x) for x in lines] if lines else []

    def latest(self) -> VRFrame | None:
        # Return an atomic snapshot so a control tick can't read a half-updated
        # frame (left hand from sample N, right from N+1) while the handler writes.
        with self._lock:
            return dataclasses.replace(self._frame, hands=dict(self._frame.hands))

    def start(self) -> None:
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        # Vuer/uvicorn doesn't expose a clean stop here; the daemon thread dies
        # with the process. Torque release is the supervisor's job.
        pass

    def _update_hand(self, side: str, mats, state) -> None:
        # Vuer HAND_MOVE: `mats` = 25 joints x 16 (each a column-major 4x4),
        # W3C XRHand order, poses relative to the wrist. tracked only when present.
        if mats is None or len(mats) < 25 * 16:
            with self._lock:
                self._frame = dataclasses.replace(
                    self._frame, stamp=time.monotonic(),
                    hands={**self._frame.hands, side: HandSample(tracked=False)})
            return
        a = np.asarray(mats, dtype=float).reshape(25, 16)
        landmarks = a[:, 12:15].copy()                  # per-joint translation (cols 12-14)
        wrist = a[0].reshape(4, 4, order="F")           # joint 0 transform
        pinch = float(state.get("pinchValue", 0.0)) if isinstance(state, dict) else 0.0
        with self._lock:
            self._frame = dataclasses.replace(
                self._frame, stamp=time.monotonic(),
                hands={**self._frame.hands, side: HandSample(
                    tracked=True, wrist=wrist, landmarks=landmarks, pinch=pinch)})

    def _op_rotation(self, side: str):
        """Operator hand ORIENTATION (3x3, WebXR) the mapping consumes, or None."""
        with self._lock:
            h = self._frame.hands.get(side)
        if h is None or not h.tracked:
            return None
        return np.asarray(h.wrist, float).reshape(4, 4)[:3, :3]

    def _calib_banner(self):
        """Build the in-headset calibration banner (a Billboard of Text lines) from
        the latest status, or an empty Billboard to clear it. Big message + a huge
        countdown + an ASCII progress bar + per-hand tracked flags, always facing you."""
        from vuer.schemas import Billboard, Text
        with self._lock:
            cal = dict(self._calib) if self._calib else None
        if not cal:
            return Billboard(key="calib", position=[0.0, 1.5, -1.0])
        phase = cal.get("phase", "wait")
        prog = float(cal.get("progress", 0.0))
        rem = float(cal.get("remaining", 0.0))
        if phase == "done":
            col, big, num = "#a6e3a1", str(cal.get("msg", "CALIBRATED")), "OK"
        elif phase == "hold":
            col, big, num = "#f9e2af", "HOLD STILL — ARMS AT YOUR SIDES", f"{rem:0.0f}"
        else:
            col, big, num = "#89b4fa", str(cal.get("msg", "DROP YOUR ARMS TO YOUR SIDES")), ""
        n = int(round(prog * 12))
        bar = "[" + "#" * n + "-" * (12 - n) + "]"
        lt = "OK" if cal.get("left") else ".."
        rt = "OK" if cal.get("right") else ".."
        return Billboard(
            Text(big, key="cal_big", position=[0.0, 0.20, 0.0], fontSize=0.062,
                 color=col, anchorX="center", anchorY="middle"),
            Text(num, key="cal_num", position=[0.0, 0.02, 0.0], fontSize=0.22,
                 color=col, anchorX="center", anchorY="middle"),
            Text(bar, key="cal_bar", position=[0.0, -0.13, 0.0], fontSize=0.05,
                 color=col, anchorX="center", anchorY="middle", font="monospace"),
            Text(f"hands    L {lt}     R {rt}", key="cal_hands", position=[0.0, -0.22, 0.0],
                 fontSize=0.04, color="#cdd6f4", anchorX="center", anchorY="middle"),
            key="calib", position=[0.0, 1.5, -1.0])

    def _hud_panel(self):
        """Build the always-on in-headset status/log panel: a left-anchored Billboard
        of monospace Text lines, pinned to the upper-left of your view. Empty -> clears."""
        from vuer.schemas import Billboard, Text
        with self._lock:
            lines = list(self._hud)
        if not lines:
            return Billboard(key="hud", position=[-0.62, 1.62, -1.0])
        children = []
        y = 0.0
        for i, ln in enumerate(lines[:9]):           # cap so it stays readable
            color = "#a6e3a1" if " OK" in ln or "TRK" in ln else (
                "#f38ba8" if "LOST" in ln else "#cdd6f4")
            children.append(Text(ln, key=f"hud_{i}", position=[0.0, y, 0.0],
                                 fontSize=0.045, color=color, anchorX="left",
                                 anchorY="middle", font="monospace"))
            y -= 0.072
        return Billboard(*children, key="hud", position=[-0.62, 1.62, -1.0])

    def _serve(self) -> None:  # pragma: no cover - needs vuer + a headset
        from vuer import Vuer
        from vuer.schemas import Hands, CoordsMarker

        if self.tunnel:
            app = Vuer(host="127.0.0.1")                                  # http; tunnel adds https
        else:
            app = Vuer(cert=self.cert, key=self.key, host="0.0.0.0")      # https on the LAN
        self._app = app

        @app.add_handler("HAND_MOVE")
        async def on_hand(event, session):
            val = event.value or {}
            self._dbg["hand"] += 1
            lk = val.get("left")
            self._dbg["last"] = f"keys={list(val.keys())} left={len(lk) if lk is not None else None}"
            self._update_hand("left", val.get("left"), val.get("leftState"))
            self._update_hand("right", val.get("right"), val.get("rightState"))

        @app.add_handler("CAMERA_MOVE")
        async def on_cam(event, session):
            self._dbg["cam"] += 1
            cam = (event.value or {}).get("camera", {})
            if cam.get("matrix") is not None:
                with self._lock:
                    self._frame = dataclasses.replace(self._frame, head=_mat4(cam["matrix"]))

        @app.add_handler("CONTROLLER_MOVE")
        async def on_ctrl(event, session):
            self._dbg["ctrl"] += 1

        @app.spawn(start=True)
        async def main(session, fps=72):
            print("[vuer] >>> QUEST CONNECTED — entered the page <<<", flush=True)
            session.upsert @ Hands(stream=True, key="hands")
            import asyncio
            tick = 0
            while True:
                await asyncio.sleep(1.0 / 30)
                # In-headset frame viz: four triads FLOATING IN FRONT of you at
                # face height (the hand poses are wrist-relative, so a triad pinned
                # to the wrist sits on the floor and vanishes in VR). Per side, YOUR
                # hand frame is the upper triad, the ROBOT's hand frame the lower —
                # compare their orientation to read the mapping. (X=red Y=green Z=blue.)
                for side in SIDES:
                    op = self._op_rotation(side)
                    if op is not None:
                        session.upsert @ CoordsMarker(
                            key=f"op_{side}", matrix=_coords_mat(_VIZ_POS[f"op_{side}"], op),
                            scale=0.13, headScale=1.6)
                    with self._lock:
                        rb = self._robot_viz.get(side)
                    if rb is not None:
                        session.upsert @ CoordsMarker(
                            key=f"rb_{side}", matrix=_coords_mat(_VIZ_POS[f"rb_{side}"], rb),
                            scale=0.13, headScale=1.6)
                session.upsert @ self._calib_banner()   # in-headset calibration countdown
                session.upsert @ self._hud_panel()      # in-headset live status/log panel
                tick += 1
                if self.debug and tick % 60 == 0:
                    d = self._dbg
                    print(f"[vuer] head_msgs={d['cam']} hand_msgs={d['hand']} "
                          f"controller_msgs={d['ctrl']} | last_hand: {d['last'] or '(none)'}", flush=True)
