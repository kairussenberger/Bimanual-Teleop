"""Record -> save -> load -> replay round-trip (spec Section 7, Replay mode).

The live-capture path needs the headset (UNVERIFIED this run); the machinery that
makes replay deterministic is tested here on synthetic frames.

    uv run pytest tests/test_replay.py -q
"""
from __future__ import annotations

import numpy as np

from bimanual_teleop.config import SIDES
from bimanual_teleop.vr.ingest import FakeVRSource
from bimanual_teleop.vr.replay import ReplaySource, SessionRecorder


def _record(n=30, hz=60.0) -> SessionRecorder:
    src = FakeVRSource()
    rec = SessionRecorder()
    for i in range(n):
        t = i / hz
        eng = {"left": i % 2 == 0, "right": i > 10}
        rec.add(src.frame_at(t), eng, t)
    return rec


def test_replay_roundtrip_matches_recording(tmp_path):
    rec = _record()
    path = tmp_path / "session.npz"
    rec.save(str(path))
    rs = ReplaySource(str(path))
    assert len(rs) == len(rec)
    src = FakeVRSource()
    for i in range(len(rec)):
        t = i / 60.0
        orig = src.frame_at(t)
        got = rs.frame_at(t)
        for s in SIDES:
            assert got.hands[s].tracked == orig.hands[s].tracked
            assert np.allclose(got.hands[s].wrist, orig.hands[s].wrist, atol=1e-6)
            assert np.allclose(got.hands[s].landmarks, orig.hands[s].landmarks, atol=1e-6)
        assert rs.engaged_at(t) == {"left": i % 2 == 0, "right": i > 10}


def test_replay_from_recorder_no_file():
    rs = ReplaySource.from_recorder(_record())
    assert len(rs) == 30
    assert abs(rs.duration - 29 / 60.0) < 1e-9
    f0 = rs.frame_at(-5.0)                     # clamp below start
    assert f0.hands["left"].tracked


def test_replay_none_landmarks_survive_roundtrip(tmp_path):
    """A hand with no landmarks (e.g. controller-only) must come back as None, not
    an array of NaNs."""
    from bimanual_teleop.vr.frames import HandSample, VRFrame
    rec = SessionRecorder()
    fr = VRFrame(stamp=0.0, head=np.eye(4),
                 hands={"left": HandSample(tracked=True, wrist=np.eye(4), landmarks=None),
                        "right": HandSample(tracked=False, wrist=np.eye(4), landmarks=None)})
    rec.add(fr, {"left": True, "right": False}, 0.0)
    p = tmp_path / "s.npz"
    rec.save(str(p))
    got = ReplaySource(str(p)).frame_at(0.0)
    assert got.hands["left"].landmarks is None
    assert got.hands["left"].tracked and not got.hands["right"].tracked


def test_replay_preserves_wrist_rotation(tmp_path):
    """A non-identity wrist ORIENTATION must survive the round-trip (not just the
    translation) — orientation is the whole point of the teleop bug."""
    from bimanual_teleop.vr.frames import HandSample, VRFrame, euler_to_R
    R = euler_to_R([0.4, -0.7, 1.2])
    wrist = np.eye(4)
    wrist[:3, :3] = R
    wrist[:3, 3] = [0.1, -0.2, 0.3]
    rec = SessionRecorder()
    rec.add(VRFrame(stamp=0.0, head=np.eye(4),
                    hands={"left": HandSample(tracked=True, wrist=wrist, landmarks=None),
                           "right": HandSample(tracked=True, wrist=np.eye(4), landmarks=None)}),
            {"left": True, "right": True}, 0.0)
    p = tmp_path / "rot.npz"
    rec.save(str(p))
    got = ReplaySource(str(p)).frame_at(0.0).hands["left"].wrist
    assert np.allclose(got, wrist, atol=1e-9)


def test_replay_loop_wraps():
    rs = ReplaySource.from_recorder(_record(n=10, hz=10.0), loop=True)
    assert abs(rs.duration - 0.9) < 1e-9
    t0 = 0.35                                   # mid-bin (avoids float boundary)
    a = rs.frame_at(t0)
    b = rs.frame_at(t0 + rs.duration)           # one full loop later -> same sample
    assert np.allclose(a.hands["left"].wrist, b.hands["left"].wrist, atol=1e-9)
    c = rs.frame_at(0.75)                        # different time -> different wrist
    assert not np.allclose(a.hands["left"].wrist, c.hands["left"].wrist, atol=1e-3)
