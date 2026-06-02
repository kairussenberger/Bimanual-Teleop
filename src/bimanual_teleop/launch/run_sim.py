"""Run the full teleop pipeline against the MuJoCo sim, in ONE process.

Single-process is the right shape for the sim demo (the MuJoCo passive viewer must
own the main thread on macOS): a background VR source feeds latest poses; the main
loop runs IK + finger retarget for both sides and steps the viewer. The
hardware-grade multi-process/ZMQ split (for the 250 Hz CAN loops on Linux) reuses
the same TeleopEngine + controllers — see launch/run_hw.py.

    uv run mjpython -m bimanual_teleop.launch.run_sim                 # viewer, fake VR
    uv run mjpython -m bimanual_teleop.launch.run_sim --vr vuer       # viewer, real Quest
    uv run python    -m bimanual_teleop.launch.run_sim --gif out.gif  # headless GIF (no window)
"""
from __future__ import annotations

import argparse
import time

from ..config import SIDES, load_rig
from ..engine import TeleopEngine
from ..safety.clutch import AlwaysOn
from ..safety.supervisor import Supervisor
from ..sim.sim_world import SimWorld
from ..vr.ingest import make_source


def run_gif(args) -> int:
    import imageio.v3 as iio
    rig = load_rig()
    if args.vr:
        rig["vr"]["transport"] = args.vr
    world = SimWorld(rig)
    engine = TeleopEngine(rig, world)
    supervisor = Supervisor(rig, AlwaysOn())
    src = make_source(rig)
    frames = []
    n = int(args.seconds * args.fps)
    for i in range(n):
        t = i / args.fps
        frame = src.frame_at(t) if hasattr(src, "frame_at") else src.latest()
        engine.tick(frame, supervisor.update(frame, t), t)
        world.step(4)
        if i % max(1, int(args.fps / 20)) == 0:
            frames.append(world.render_rgb(azimuth=args.azimuth, elevation=args.elevation))
    iio.imwrite(args.gif, frames, duration=1000 / 20, loop=0)
    print(f"wrote {args.gif} ({len(frames)} frames)")
    return 0


def run_viewer(args) -> int:
    import mujoco.viewer
    rig = load_rig()
    if args.vr:
        rig["vr"]["transport"] = args.vr
    world = SimWorld(rig)
    engine = TeleopEngine(rig, world)
    supervisor = Supervisor(rig, AlwaysOn())
    src = make_source(rig)
    src.start()
    try:
        with mujoco.viewer.launch_passive(world.model, world.data) as v:
            while v.is_running():
                t = time.monotonic()   # one clock shared with the source stamps + supervisor
                frame = src.latest()
                engine.tick(frame, supervisor.update(frame, t), t)
                world.step(2)
                v.sync()
                time.sleep(1 / 120)
    finally:
        src.stop()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vr", choices=["fake", "vuer"], help="override vr.transport")
    ap.add_argument("--gif", metavar="PATH", help="headless: render a GIF and exit")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--azimuth", type=float, default=70)
    ap.add_argument("--elevation", type=float, default=-18)
    args = ap.parse_args()
    return run_gif(args) if args.gif else run_viewer(args)


if __name__ == "__main__":
    raise SystemExit(main())
