"""Run the teleop pipeline against REAL hardware (Linux control host).

This is the hardware bring-up entrypoint. It reuses the exact TeleopEngine +
controllers + supervisor as the sim — only the *sink* changes (HardwareSink →
YAM CAN + ORCA serial). Single-process bring-up form; for production split each
arm into its own ~250 Hz CAN process (see README "Hardware day").

Prereqs (Ubuntu): SocketCAN up (`sudo ip link set can0 up type can bitrate 1000000`,
same for can1), i2rt SDK installed, ORCA hands tensioned + calibrated.

    python -m bimanual_teleop.launch.run_hw --vr vuer

SAFETY: starts in IDLE (not following). Engage via the configured clutch. Ctrl+C
or e-stop releases torque on all devices.
"""
from __future__ import annotations

import argparse
import sys
import time

from ..config import SIDES, load_rig
from ..engine import TeleopEngine
from ..safety.clutch import GestureClutch
from ..safety.supervisor import Supervisor
from ..vr.ingest import make_source


def main() -> int:
    if sys.platform == "darwin":
        print("WARNING: real YAM control needs Linux/SocketCAN; macOS can't run the CAN loop.")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vr", choices=["vuer", "fake"], default="vuer")
    ap.add_argument("--hz", type=float, default=None, help="override control rate")
    args = ap.parse_args()

    rig = load_rig()
    rig["vr"]["transport"] = args.vr
    hz = args.hz or rig["control"]["arm_hz"]

    from ..hardware import HardwareSink
    sink = HardwareSink(rig)
    engine = TeleopEngine(rig, sink)
    supervisor = Supervisor(rig, GestureClutch())
    src = make_source(rig)
    src.start()

    period = 1.0 / hz
    try:
        while True:
            t = time.monotonic()   # shared clock with source stamps + supervisor staleness
            frame = src.latest()
            engine.tick(frame, supervisor.update(frame, t), t)
            dt = period - (time.monotonic() - t)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        print("\nstopping — releasing torque")
    finally:
        supervisor.estop()
        src.stop()
        sink.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
