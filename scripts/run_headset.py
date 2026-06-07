#!/usr/bin/env python
"""Windowless WebXR teleop server WITH the in-headset HUD — no MuJoCo window needed.

Serves the Vuer page and runs the teleop engine in a windowless sim, so the operator
opens the page in the Quest browser, enters VR, and gets IN-HEADSET feedback: the
status/log panel (tracking flags, wrist roll, clutch/calib state, loop rate) + the
hand/robot frame triads — exactly what ORBIT can't show.

Connection is over USB (adb reverse), so no Wi-Fi or cert juggling:
  - serves HTTPS on :8012 (cert.pem/key.pem) on all interfaces,
  - `adb reverse tcp:8012` maps the Quest's localhost:8012 to this machine,
  - open  https://localhost:8012  in the Quest browser (accept the cert warning) →
    https on localhost is a WebXR secure context, so immersive VR works.
  - same-Wi-Fi fallback: https://<this-LAN-IP>:8012

    uv run python scripts/run_headset.py
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bimanual_teleop.config import SIDES, load_rig                       # noqa: E402
from bimanual_teleop.engine import TeleopEngine                         # noqa: E402
from bimanual_teleop.launch.run_sim import _hud_lines, _push_robot_frames  # noqa: E402
from bimanual_teleop.logging_utils import RateMeter, get_logger         # noqa: E402
from bimanual_teleop.safety.clutch import AlwaysOn                      # noqa: E402
from bimanual_teleop.safety.supervisor import Supervisor               # noqa: E402
from bimanual_teleop.sim.sim_world import SimWorld                      # noqa: E402
from bimanual_teleop.vr.vuer_source import VuerVRSource                # noqa: E402

log = get_logger("headset")


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "<LAN-IP>"
    finally:
        s.close()


def main() -> int:
    rig = load_rig()
    rig["vr"]["transport"] = "vuer"
    # HTTPS on 0.0.0.0 with cert.pem/key.pem. Must be HTTPS end-to-end: vuer's client
    # is https, so an insecure ws to localhost gets killed as mixed content (the
    # "connected then immediately disconnected" bug). Over adb-reverse the Quest opens
    # https://localhost:8012 (one cert-accept tap) → secure wss → stable WebXR.
    rig["vr"]["tunnel"] = False
    rig["vr"]["debug"] = True             # log head/hand msg counts so we can see tracking

    # USB path: forward the Quest's localhost:8012 to this machine (no Wi-Fi needed).
    try:
        subprocess.run(["adb", "reverse", "tcp:8012", "tcp:8012"], capture_output=True, timeout=5)
        log.info("adb reverse tcp:8012 set (USB path ready)")
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        log.warning("adb reverse failed — use the Wi-Fi URL instead")

    world = SimWorld(rig)
    engine = TeleopEngine(rig, world)
    sup = Supervisor(rig, AlwaysOn())
    src = VuerVRSource(rig)
    src.start()

    ip = _lan_ip()
    print("\n" + "=" * 64
          + "\n  OPEN ON THE QUEST BROWSER, then press 'Enter VR':\n"
          f"    USB:  https://localhost:8012   (tap Advanced -> Proceed on the cert warning)\n"
          f"    Wi-Fi: https://{ip}:8012\n"
          "  You'll see a status/log panel + hand triads floating in front of you.\n"
          "  First: drop both arms to your sides ~5s for calibration.\n"
          + "=" * 64 + "\n", flush=True)
    _ = ip

    rate = RateMeter()
    t_prev = None
    try:
        while True:
            t = time.monotonic()
            if t_prev is not None:
                rate.update(t - t_prev)
            t_prev = t
            frame = src.latest()
            eng = sup.update(frame, t)
            engine.tick(frame, eng, t)
            src.set_calib(engine.calib_status)
            src.set_hud(_hud_lines(engine, frame, eng, rate.hz))
            if frame is not None:
                _push_robot_frames(src, engine, frame)
            world.step(2)
            time.sleep(1.0 / 120.0)
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
