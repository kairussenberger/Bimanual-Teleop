"""TeleopViz (local Rerun 3D viewer) smoke — runs the real engine + fake VR source
through the scene writer and checks an .rrd actually lands on disk. Skipped when
the optional rerun-sdk extra is not installed (headless CI stays lean)."""
from __future__ import annotations

import pytest


class _NullSink:
    def set_arm(self, side, q):
        pass

    def set_hand(self, side, joints):
        pass


def test_teleop_viz_logs_rrd(tmp_path):
    pytest.importorskip("rerun")
    from bimanual_teleop.config import SIDES, load_rig
    from bimanual_teleop.engine import TeleopEngine
    from bimanual_teleop.viz.live_viz import TeleopViz
    from bimanual_teleop.vr.ingest import FakeVRSource

    rig = load_rig()
    rig["vr"]["calib_seconds"] = 0
    out = tmp_path / "scene.rrd"
    viz = TeleopViz(rig, spawn=False, save_path=str(out))
    engine = TeleopEngine(rig, _NullSink())
    src = FakeVRSource()
    for i in range(30):
        t = i / 60.0
        frame = src.frame_at(t)
        engaged = {s: True for s in SIDES}
        engine.tick(frame, engaged, t)
        viz.tick(engine=engine, frame=frame, engaged=engaged, hz=60.0, t=t)
    try:                                  # flush stream buffers before sizing the file
        viz.log.rr.flush(blocking=True)
    except Exception:
        pass
    assert out.exists() and out.stat().st_size > 1000
