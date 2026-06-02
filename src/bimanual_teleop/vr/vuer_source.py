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
        self._lock = threading.Lock()
        self._frame = VRFrame(hands={s: HandSample() for s in SIDES})
        self._thread: threading.Thread | None = None
        self._app = None

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

    def _serve(self) -> None:  # pragma: no cover - needs vuer + a headset
        from vuer import Vuer
        from vuer.schemas import Hands

        if self.tunnel:
            app = Vuer(host="127.0.0.1")                                  # http; tunnel adds https
        else:
            app = Vuer(cert=self.cert, key=self.key, host="0.0.0.0")      # https on the LAN
        self._app = app

        @app.add_handler("HAND_MOVE")
        async def on_hand(event, session):
            val = event.value or {}
            if self.debug:
                lk = val.get("left")
                print("HAND_MOVE keys:", list(val.keys()),
                      "| left floats:", (len(lk) if lk is not None else None), flush=True)
            self._update_hand("left", val.get("left"), val.get("leftState"))
            self._update_hand("right", val.get("right"), val.get("rightState"))

        @app.add_handler("CAMERA_MOVE")
        async def on_cam(event, session):
            cam = (event.value or {}).get("camera", {})
            if cam.get("matrix") is not None:
                with self._lock:
                    self._frame = dataclasses.replace(self._frame, head=_mat4(cam["matrix"]))

        @app.spawn(start=True)
        async def main(session, fps=72):
            session.upsert @ Hands(stream=True, key="hands")
            import asyncio
            while True:
                await asyncio.sleep(1.0)

        app.run()
