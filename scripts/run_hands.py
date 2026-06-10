#!/usr/bin/env python
"""Run ONLY the ORBIT hand-tracking browser viewer — no MuJoCo, no robot.

Starts the ORBIT source (adb reverse + NetMQ ingest) and serves the live hand
viewer at http://127.0.0.1:8099, where you calibrate and see the two wrist
vectors. Stays up until Ctrl-C. The MuJoCo sim is NOT loaded.

    # 1) launch the headset app:
    adb shell monkey -p com.ORBIT.Teleoperation -c android.intent.category.LAUNCHER 1
    # 2) run the viewer:
    uv run python scripts/run_hands.py
    uv run python scripts/run_hands.py --no-browser   # don't auto-open a tab
"""
from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from bimanual_teleop.config import load_rig            # noqa: E402
from bimanual_teleop.vr.orbit_source import OrbitVRSource  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-browser", action="store_true", help="don't auto-open the viewer tab")
    args = ap.parse_args()

    rig = load_rig()
    src = OrbitVRSource(rig)          # binds the ORBIT ports + serves the browser viz at :8099
    src.start()
    url = getattr(src, "viz_url", None) or "http://127.0.0.1:8099"
    print("\n" + "=" * 60 + f"\n  ORBIT hands viewer (no MuJoCo): {url}\n"
          "  Wear the headset, set BOTH controllers down (hand-tracking only).\n"
          "  Ctrl-C to stop.\n" + "=" * 60 + "\n", flush=True)
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nstopping…", flush=True)
    finally:
        src.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
