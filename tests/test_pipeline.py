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
    T0 = ik.fk_wrist()                 # position IK targets the WRIST site
    # Move UP+forward into the workspace — the direction teleop actually drives from
    # the arms-down home. (A target further DOWN sits near the hanging arm's reach
    # boundary, where any IK is stiff; that's not what this convergence test checks.)
    tgt = T0.translation() + np.array([0.07, 0.0, 0.05])
    target = mink.SE3.from_rotation_and_translation(ik.fk_ee().rotation(), tgt)
    for _ in range(300):
        ik.solve(target)
    assert np.linalg.norm(ik.fk_wrist().translation() - tgt) < 5e-3


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


def test_clutch_release_disengages_immediately():
    """Releasing the deadman on a LIVE feed must stop following at once (the HOLD
    window is only for tracking dropouts) — the deadman bug the review caught."""
    from bimanual_teleop.config import load_rig, SIDES
    from bimanual_teleop.safety.supervisor import Supervisor
    from bimanual_teleop.safety.clutch import KeyboardClutch
    from bimanual_teleop.vr.frames import VRFrame, HandSample
    rig = load_rig()
    kc = KeyboardClutch()
    sup = Supervisor(rig, kc)
    fr = VRFrame(stamp=100.0, hands={s: HandSample(tracked=True) for s in SIDES})
    kc.held = True
    assert sup.update(fr, 100.0)["left"]
    kc.held = False
    fr2 = VRFrame(stamp=100.05, hands={s: HandSample(tracked=True) for s in SIDES})
    assert not sup.update(fr2, 100.05)["left"]   # immediate, NOT after hold_s


def test_abs_orientation_no_engage_snap():
    """abs-orientation mode must equal the anchored EE pose at the engage instant
    (the ~157° wrist snap the review caught)."""
    import mink
    from bimanual_teleop.vr.frames import ClutchMapper, euler_to_R
    R_ee = euler_to_R([0.5, -0.3, 0.8])
    R_ctrl = euler_to_R([1.1, 0.2, -0.4])
    ee = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R_ee), np.array([0.3, 0.1, 0.5]))
    ctrl = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R_ctrl), np.array([1.0, 2.0, 3.0]))
    m = ClutchMapper(euler_to_R([0.2, 0.0, 1.5]), pos_scale=1.0, abs_orientation=True)
    m.engage(ctrl, ee)
    tgt = m.target(ctrl)
    assert np.allclose(tgt.rotation().as_matrix(), R_ee, atol=1e-9)
    assert np.allclose(tgt.translation(), ee.translation(), atol=1e-9)


def test_calibration_aligns_forward_and_no_cross():
    """Reference-stance calibration → 'hand forward' maps to robot forward (−X),
    and the anti-cross clamp keeps each hand on its own side of center."""
    from bimanual_teleop.config import load_rig
    from bimanual_teleop.arms.arm_control import ArmController
    from bimanual_teleop.vr.calibrate import calibrate_R
    from bimanual_teleop.vr.frames import HandSample, quat_to_R
    lm = np.zeros((25, 3))                              # fingers forward (−z), palm down
    lm[6] = [0.03, 0, -0.03]; lm[21] = [-0.03, 0, -0.03]
    lm[9] = [0.03, 0, -0.15]; lm[14] = [0, 0, -0.16]; lm[19] = [-0.01, 0, -0.15]
    wm = lambda p: np.block([[np.eye(3), np.array(p).reshape(3, 1)], [0, 0, 0, 1]])
    rig = load_rig()
    for side, sign in (("left", -1), ("right", +1)):
        ac = ArmController(rig, side)
        ac.mapper.set_R(calibrate_R(lm, rig["arms"][side]["base_quat"]))
        bR = quat_to_R(rig["arms"][side]["base_quat"]); bp = np.array(rig["arms"][side]["base_pos"])
        wp = lambda: bR @ ac.ik.fk_wrist().translation() + bp     # wrist position in world (the IK target)
        ac.ik.reset(); t = 0.0
        ac.update(HandSample(tracked=True, wrist=wm([0, 0, 0]), landmarks=lm), True, t)
        x0 = wp()[0]
        for _ in range(120):            # from the arms-down home the arm must swing
            t += 1 / 120                # toward forward first, so it needs more ticks
            ac.update(HandSample(tracked=True, wrist=wm([0, 0, -0.25]), landmarks=lm), True, t)
        assert wp()[0] < x0 - 0.02      # forward = −X
        # shove toward center; must stay on own side
        ac.ik.reset(); t = 0.0
        ac.update(HandSample(tracked=True, wrist=wm([0, 0, 0]), landmarks=lm), True, t)
        for _ in range(80):
            t += 1 / 120
            ac.update(HandSample(tracked=True, wrist=wm([-sign * 0.5, 0, 0]), landmarks=lm), True, t)
        y = wp()[1]
        assert (y <= 0.001) if side == "left" else (y >= -0.001)


def test_end_to_end_sim_tick():
    """Fake VR → engine → sim moves the arms (EE position changes)."""
    from bimanual_teleop.config import load_rig, SIDES
    from bimanual_teleop.engine import TeleopEngine
    from bimanual_teleop.sim.sim_world import SimWorld
    from bimanual_teleop.vr.ingest import FakeVRSource
    rig = load_rig()
    rig["vr"]["calib_seconds"] = 0          # skip the calibration phase for this motion test
    world = SimWorld(rig)
    engine = TeleopEngine(rig, world)
    src = FakeVRSource()
    ee0 = world.ee_pose("left")[:3, 3].copy()
    for i in range(120):
        t = i / 60.0
        engine.tick(src.frame_at(t), {s: True for s in SIDES}, t)
        world.step(2)
    assert np.linalg.norm(world.ee_pose("left")[:3, 3] - ee0) > 0.02


def test_overlay_geom_counts_and_guard():
    """Overlay primitives append the right number of geoms and never overflow."""
    import mujoco
    from bimanual_teleop.viz import overlay
    m = mujoco.MjModel.from_xml_string("<mujoco/>")
    scn = mujoco.MjvScene(m, maxgeom=1000); scn.ngeom = 0
    overlay.triad(scn, [0, 0, 0], np.eye(3));                    assert scn.ngeom == 3
    overlay.sphere(scn, [0, 0, 0], 0.01, (1, 1, 1, 1));         assert scn.ngeom == 4
    overlay.connector(scn, [0, 0, 0], [0, 0, 1], 0.01, (1, 1, 1, 1)); assert scn.ngeom == 5
    overlay.skeleton(scn, np.zeros((25, 3)))                     # 24 bones + 25 joints
    assert scn.ngeom == 5 + len(overlay.HAND_BONES) + 25
    n = scn.ngeom
    overlay.arrow(scn, [0, 0, 0], [0, 0, 0], 0.1, 0.01, (1, 1, 1, 1))  # zero dir = no-op
    assert scn.ngeom == n
    full = mujoco.MjvScene(m, maxgeom=2); full.ngeom = 0
    overlay.skeleton(full, np.zeros((25, 3)))                    # must respect maxgeom
    assert full.ngeom <= 2


def test_studio_canonical_hand_axes():
    """The synthetic operator's reference hand reads forward=−z (fingers), up=+y."""
    from bimanual_teleop.tools import mapping_studio as ms
    from bimanual_teleop.vr.calibrate import operator_axes
    ax = operator_axes(ms._canonical_hand())     # columns [right, up, forward]
    assert ax[2, 2] < -0.9      # forward's z-component ≈ −1
    assert ax[1, 1] > 0.9       # up's y-component ≈ +1


def test_studio_synthetic_hand_consistent_with_wrist():
    """Synthetic landmarks must ride the wrist orientation (the reason the studio
    has its own driver instead of FakeVRSource): hand_frame(lm) == R_wrist · hand_frame(ref)."""
    from bimanual_teleop.tools import mapping_studio as ms
    from bimanual_teleop.hands.quest_retarget import hand_frame
    _, Hc = hand_frame(ms._CANON)
    for t in (0.7, 1.9, 3.3):
        h = ms._synthetic_frame(t).hands["left"]
        R = np.asarray(h.wrist, float)[:3, :3]
        _, Hf = hand_frame(h.landmarks)
        assert np.allclose(Hf, R @ Hc, atol=1e-6)


def test_studio_tuner_retunes_mapping():
    """Live knobs nudge R_base_from_vr + pos_scale on both arms; reset restores."""
    from bimanual_teleop.config import load_rig
    from bimanual_teleop.engine import TeleopEngine
    from bimanual_teleop.sim.sim_world import SimWorld
    from bimanual_teleop.tools.mapping_studio import Tuner
    rig = load_rig(); rig["vr"]["calib_seconds"] = 0
    world = SimWorld(rig); engine = TeleopEngine(rig, world)
    tn = Tuner(rig, engine); tn.apply()
    R0 = engine.arm["left"].mapper.R.copy()
    tn.key(ord("I")); tn.key(ord("I")); tn.apply()              # +pitch tweak
    assert not np.allclose(engine.arm["left"].mapper.R, R0)
    tn.key(ord("0")); tn.apply()                                # reset
    assert np.allclose(engine.arm["left"].mapper.R, R0, atol=1e-9)
    tn.key(ord("=")); tn.apply()                                # scale up one step
    assert abs(engine.arm["left"].mapper.scale
               - (rig["mapping"]["pos_scale"] + Tuner.SCALE_STEP)) < 1e-9


def test_studio_tuner_reengages_after_retune():
    """Retuning the frame mid-session (set_R → release) must re-anchor on the next
    engaged tick, not crash on the missing clutch anchor (the live-knob bug the
    adversarial review caught)."""
    from bimanual_teleop.config import load_rig, SIDES
    from bimanual_teleop.engine import TeleopEngine
    from bimanual_teleop.sim.sim_world import SimWorld
    from bimanual_teleop.tools.mapping_studio import Tuner, _synthetic_frame
    rig = load_rig(); rig["vr"]["calib_seconds"] = 0
    world = SimWorld(rig); engine = TeleopEngine(rig, world)
    tn = Tuner(rig, engine)
    for i in range(20):                                   # engage + follow
        t = i / 60.0
        engine.tick(_synthetic_frame(t), {s: True for s in SIDES}, t); world.step(1)
    assert engine.arm["left"].mapper.engaged
    tn.key(ord("I")); tn.apply()                          # retune → set_R releases clutch
    assert not engine.arm["left"].mapper.engaged
    ee0 = world.ee_pose("left")[:3, 3].copy()
    for i in range(20, 60):                               # must re-anchor, not assert-crash
        t = i / 60.0
        engine.tick(_synthetic_frame(t), {s: True for s in SIDES}, t); world.step(1)
    assert engine.arm["left"].mapper.engaged              # re-anchored
    assert np.linalg.norm(world.ee_pose("left")[:3, 3] - ee0) > 0.005   # still following


def test_orbit_source_unity_to_webxr():
    """ORBIT (Unity, +z fwd, LH) -> engine (WebXR, -z fwd, RH) is the single Z-flip
    congruence: Palm[1] dropped (26->25 W3C joints), wrist pose converted, staleness honored."""
    import time
    from bimanual_teleop.vr.orbit_source import OrbitVRSource
    src = OrbitVRSource({"vr": {"orbit_flip": "z", "orbit_adb_reverse": False, "orbit_timeout": 5.0}})
    pts = np.zeros((26, 3)); pts[:, 2] = np.arange(26) * 0.01; pts[1] = [9.0, 9.0, 9.0]  # Palm sentinel
    hand_msg = "relative:" + "|".join(f"{x},{y},{z}" for x, y, z in pts) + ":"
    now = time.monotonic()
    src._ingest("hand", "right", hand_msg, now)
    src._ingest("wrist", "right", "relative,0.1,0.2,0.3,0,0,0,1", now)
    hr = src.latest().hands["right"]
    assert hr.tracked and hr.landmarks.shape == (25, 3)               # tracked, palm dropped
    assert not np.allclose(hr.landmarks[1], [9.0, 9.0, -9.0])         # the Palm sentinel is gone
    assert np.isclose(hr.landmarks[1, 2], -0.02)                     # orig joint 2, z negated
    assert np.allclose(hr.wrist[:3, 3], [0.1, 0.2, -0.3], atol=1e-6)  # wrist z flipped
    src._last["right"] = now - 10.0                                  # staleness -> not tracked
    assert not src.latest().hands["right"].tracked


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
