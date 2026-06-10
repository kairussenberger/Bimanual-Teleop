from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_check_quest():
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_quest.py"
    spec = importlib.util.spec_from_file_location("check_quest", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_check_roll():
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_roll.py"
    spec = importlib.util.spec_from_file_location("check_roll", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_quest_head_ok_handles_missing_head():
    cq = _load_check_quest()
    assert cq._head_ok(None) is False
    assert cq._head_ok(np.eye(4)) is False
    head = np.eye(4)
    head[:3, 3] = [0.0, 1.6, 0.0]
    assert cq._head_ok(head) is True


def test_check_quest_formats_body_relative_wrist_vector():
    from bimanual_teleop.vr.calibrate import head_op_axes
    from bimanual_teleop.vr.frames import HandSample

    cq = _load_check_quest()
    torso_from_head = np.array([0.0, -0.35, 0.0])
    wrist_body = np.array([0.21, 0.32, 0.48])
    head = np.eye(4)
    head[:3, 3] = [0.0, 1.6, 0.0]
    op_axes = head_op_axes(head)
    wrist = np.eye(4)
    wrist[:3, :3] = op_axes
    wrist[:3, 3] = head[:3, 3] + op_axes @ (torso_from_head + wrist_body)

    text = cq._fmt_hand(HandSample(tracked=True, wrist=wrist), head, torso_from_head)
    assert "TRACKED" in text
    assert "body=[+0.210 +0.320 +0.480]" in text


def test_check_quest_formats_no_head_without_raw_fallback():
    from bimanual_teleop.vr.frames import HandSample

    cq = _load_check_quest()
    wrist = np.eye(4)
    wrist[:3, 3] = [2.0, 1.5, -1.0]
    text = cq._fmt_hand(HandSample(tracked=True, wrist=wrist), None)
    assert "TRACKED" in text
    assert "body=NO_HEAD" in text


def test_check_roll_requires_head_for_body_relative_rotation():
    from bimanual_teleop.vr.frames import HandSample, VRFrame

    cr = _load_check_roll()
    frame = VRFrame(
        stamp=0.0,
        head=None,
        hands={"right": HandSample(tracked=True, wrist=np.eye(4))},
    )
    assert cr._body_relative_wrist_rotation(frame, "right") is None


def test_check_roll_body_relative_rotation_removes_head_yaw():
    from bimanual_teleop.vr.calibrate import head_op_axes
    from bimanual_teleop.vr.frames import HandSample, VRFrame, euler_to_R

    cr = _load_check_roll()
    wrist_body_R = euler_to_R([0.4, -0.2, 0.7])
    got = []
    for yaw in (0.0, 0.8):
        head = np.eye(4)
        head[:3, :3] = euler_to_R([0.0, yaw, 0.0])
        head[:3, 3] = [0.2, 1.6, -0.1]
        wrist = np.eye(4)
        wrist[:3, :3] = head_op_axes(head) @ wrist_body_R
        frame = VRFrame(
            stamp=0.0,
            head=head,
            hands={"right": HandSample(tracked=True, wrist=wrist)},
        )
        got.append(cr._body_relative_wrist_rotation(frame, "right"))

    assert np.allclose(got[0], wrist_body_R, atol=1e-9)
    assert np.allclose(got[1], wrist_body_R, atol=1e-9)
