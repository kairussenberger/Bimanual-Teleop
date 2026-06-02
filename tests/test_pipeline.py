"""Fast, hardware-free tests for the teleop pipeline's pure logic + sim wiring.

    uv run python -m pytest tests/ -q      (or just: uv run python tests/test_pipeline.py)
"""
from __future__ import annotations

import time

import numpy as np


def test_joint_name_map_roundtrip():
    from bimanual_teleop.hands.joint_map import orca_to_sim_short, sim_short_to_orca
    for j in ["wrist", "thumb_cmc", "thumb_abd", "thumb_mcp", "thumb_dip",
              "index_abd", "index_mcp", "index_pip", "pinky_pip", "middle_mcp"]:
        assert sim_short_to_orca(orca_to_sim_short(j)) == j
    assert orca_to_sim_short("thumb_dip") == "t-pip"      # the tricky one
    assert orca_to_sim_short("index_mcp") == "i-mcp"


def test_quest_retarget_open_to_fist():
    from bimanual_teleop.hands.quest_retarget import quest_to_orca, synthetic_webxr_hand
    from bimanual_teleop.hands.joint_map import load_hand_config
    from bimanual_teleop.hands import retarget_core as rc
    neutral, roms = load_hand_config("orcahand_right")
    op = rc.clamp_to_rom(quest_to_orca(synthetic_webxr_hand(0.0), neutral, mirror=True), roms)
    fist = rc.clamp_to_rom(quest_to_orca(synthetic_webxr_hand(1.0), neutral, mirror=True), roms)
    # fingers curl more in a fist; all outputs stay within ROM
    for f in ("index", "middle", "ring", "pinky"):
        assert fist[f"{f}_mcp"] > op[f"{f}_mcp"] + 20
        assert fist[f"{f}_pip"] >= op[f"{f}_pip"]
    for j, v in fist.items():
        lo, hi = roms[j]
        assert lo - 1e-6 <= v <= hi + 1e-6


def test_clutch_mapper_relative_zero_motion_on_engage():
    import mink
    from bimanual_teleop.vr.frames import ClutchMapper
    m = ClutchMapper(np.eye(3), pos_scale=1.0, abs_orientation=False)
    ee = mink.SE3.from_translation(np.array([0.3, 0.1, 0.5]))
    ctrl = mink.SE3.from_translation(np.array([1.0, 2.0, 3.0]))
    m.engage(ctrl, ee)
    # no controller motion -> target == anchored EE
    tgt = m.target(ctrl)
    assert np.allclose(tgt.translation(), ee.translation(), atol=1e-9)
    # +5cm controller x -> +5cm EE x (scale 1, identity R)
    moved = mink.SE3.from_translation(np.array([1.05, 2.0, 3.0]))
    assert np.allclose(m.target(moved).translation(), ee.translation() + [0.05, 0, 0], atol=1e-9)


def test_arm_ik_converges():
    import mink
    from bimanual_teleop.arms.ik import ArmIK
    from bimanual_teleop.config import load_rig
    ik = ArmIK(load_rig(), "left")
    T0 = ik.fk_ee()
    tgt = T0.translation() + np.array([0.07, 0.0, -0.05])
    target = mink.SE3.from_rotation_and_translation(T0.rotation(), tgt)
    for _ in range(300):
        ik.solve(target)
    assert np.linalg.norm(ik.fk_ee().translation() - tgt) < 5e-3


def test_supervisor_estop_and_staleness():
    from bimanual_teleop.config import load_rig
    from bimanual_teleop.safety.supervisor import Supervisor
    from bimanual_teleop.safety.clutch import AlwaysOn
    from bimanual_teleop.vr.frames import VRFrame, HandSample
    rig = load_rig()
    sup = Supervisor(rig, AlwaysOn())
    fresh = VRFrame(stamp=10.0, hands={s: HandSample(tracked=True) for s in ("left", "right")})
    eng = sup.update(fresh, t=10.0)
    assert eng["left"] and eng["right"]
    # stale frame well past hold window -> not engaged
    eng = sup.update(fresh, t=10.0 + rig["safety"]["hold_s"] + 1.0)
    assert not eng["left"] and not eng["right"]
    sup.estop()
    assert not any(sup.update(fresh, t=12.0).values())


def test_zmq_loopback_latest_value():
    from bimanual_teleop.bus.zmq_io import Publisher, LatestSub
    from bimanual_teleop.bus import topics
    ep = "tcp://127.0.0.1:5799"
    pub = Publisher(ep)
    sub = LatestSub(ep, [topics.ARM_CMD])
    time.sleep(0.2)  # PUB/SUB slow-joiner
    for q in range(5):
        pub.send(topics.ARM_CMD, topics.msg(stamp=float(q), side="left", q=np.full(6, q)))
    time.sleep(0.05)
    sub.poll()
    got = sub.get(topics.ARM_CMD)
    assert got is not None and got["stamp"] == 4.0  # only the latest survives
    pub.close(); sub.close()


def test_end_to_end_sim_tick():
    """Fake VR → engine → sim moves the arms (EE position changes)."""
    from bimanual_teleop.config import load_rig, SIDES
    from bimanual_teleop.engine import TeleopEngine
    from bimanual_teleop.sim.sim_world import SimWorld
    from bimanual_teleop.vr.ingest import FakeVRSource
    rig = load_rig()
    world = SimWorld(rig)
    engine = TeleopEngine(rig, world)
    src = FakeVRSource()
    ee0 = world.ee_pose("left")[:3, 3].copy()
    for i in range(120):
        t = i / 60.0
        engine.tick(src.frame_at(t), {s: True for s in SIDES}, t)
        world.step(2)
    assert np.linalg.norm(world.ee_pose("left")[:3, 3] - ee0) > 0.02


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fail = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:
            fail += 1
            print("FAIL", fn.__name__, "->", type(e).__name__, e)
    sys.exit(1 if fail else 0)
