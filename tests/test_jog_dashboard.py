"""Keyboard jog (JogSession) + dashboard (StateFeed/server) — headless tests."""
from __future__ import annotations

import http.client
import importlib.util
import json
import threading
import uuid
from pathlib import Path

import numpy as np
import pytest

from bimanual_teleop.config import load_rig
from bimanual_teleop.render_sink import RenderSink

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def render_sink():
    rig = load_rig()
    rig["vr"]["unity_json_endpoint"] = None
    sink = RenderSink(rig, endpoint=f"inproc://jog-test-{uuid.uuid4()}")
    yield rig, sink
    sink.close()


def test_jog_joint_step_moves_only_selected_joint_within_limits(render_sink):
    rig, sink = render_sink
    jog = _load("jog_arms").JogSession(rig, sink)
    jog.side, jog.joint = "right", 5
    q0 = jog.ik["right"].q
    q1 = jog.step_joint(-1)
    assert abs((q1 - q0)[5] + jog.joint_step) < 1e-9
    assert np.allclose((q1 - q0)[:5], 0.0)
    for _ in range(500):                                   # hammer into the soft stop
        jog.step_joint(-1)
    q = jog.ik["right"].q
    assert q[5] >= jog.ik["right"].soft_lo[5] - 1e-9       # clamped, never past
    assert np.allclose(jog.ik["left"].q, jog.ik["left"].q0)  # other arm untouched


def test_jog_ee_nudge_moves_wrist_in_world_direction(render_sink):
    rig, sink = render_sink
    mod = _load("jog_arms")
    jog = mod.JogSession(rig, sink)
    jog.side = "right"
    shim = jog.engine.arm["right"]
    p0 = shim.base_R @ jog.ik["right"].fk_wrist().translation() + shim.base_pos
    for _ in range(12):
        jog.nudge_ee([-jog.ee_step, 0.0, 0.0])             # forward = world −X
    p1 = shim.base_R @ jog.ik["right"].fk_wrist().translation() + shim.base_pos
    d = p1 - p0
    assert d[0] < -0.05                                     # moved forward
    assert abs(d[0]) > max(abs(d[1]), abs(d[2]))            # dominantly forward
    jog.publish(60.0, 1.0)                                  # build_state works on the shim
    assert "right" in jog.status_line().lower() or "RIGHT" in jog.status_line()


def test_dashboard_serves_injected_state():
    mod = _load("dashboard")
    feed = mod.StateFeed("tcp://127.0.0.1:1")               # never started — injected state
    feed.latest = {"status": {"hz": 100.0, "tracked": {"left": True, "right": False},
                              "engaged": {"left": False, "right": False}, "calib": None},
                   "arms": {}, "op": {"hands": {}}}
    srv = mod.make_server(feed, "127.0.0.1", 0)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("GET", "/state")
        d = json.loads(conn.getresponse().read())
        assert d["connected"] is False
        assert d["state"]["status"]["hz"] == 100.0
        conn.request("GET", "/")
        page = conn.getresponse().read().decode()
        assert "dashboard" in page and "torso" in page
    finally:
        srv.shutdown()
        srv.server_close()
