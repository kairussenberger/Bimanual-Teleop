from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from bimanual_teleop.config import SIDES
from bimanual_teleop.bus import topics
from bimanual_teleop.hands.joint_map import ORCA_JOINT_ORDER


def _load_monitor():
    script = Path(__file__).resolve().parents[1] / "scripts" / "render_monitor.py"
    spec = importlib.util.spec_from_file_location("render_monitor", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _state() -> dict:
    arms = {}
    hand_render = {}
    op_hands = {}
    for side in SIDES:
        arms[side] = {
            "q": [0.0] * 6,
            "link_pos": [0.0] * 24,
            "ee_pos": [0.0] * 3,
            "ee_quat": [1.0, 0.0, 0.0, 0.0],
            "cmd_pos": [0.0, 0.0, 0.0],
            "cmd_quat": [1.0, 0.0, 0.0, 0.0],
        }
        hand_render[side] = {
            "names": list(ORCA_JOINT_ORDER),
            "q": [0.0] * len(ORCA_JOINT_ORDER),
        }
        op_hands[side] = {
            "tracked": True,
            "wrist_body": [0.0, 0.4, 0.2],
        }
    return {
        "v": topics.SCHEMA_VERSION,
        "arms": arms,
        "hand_render": hand_render,
        "op": {
            "torso_from_head": [0.0, -0.35, 0.0],
            "head_pos": [0.0, 1.6, 0.0],
            "torso_pos": [0.0, 1.25, 0.0],
            "hands": op_hands,
        },
        "status": {
            "engaged": {"left": True, "right": True},
            "tracked": {"left": True, "right": True},
            "hz": 100.0,
        },
    }


def test_render_monitor_bimanual_state_validation_accepts_full_payload():
    monitor = _load_monitor()
    monitor._validate_bimanual_state(_state(), require_hand_render=True)
    monitor._validate_bimanual_state(_state(), require_hand_render=True, require_command_target=True)


def test_render_monitor_prints_command_error(capsys):
    monitor = _load_monitor()
    st = _state()
    st["arms"]["right"]["ee_pos"] = [0.0, 0.0, 0.0]
    st["arms"]["right"]["cmd_pos"] = [0.03, 0.04, 0.0]
    monitor._print_state(
        st,
        require_hand_render=True,
        require_bimanual_state=True,
        require_command_target=True,
    )
    out = capsys.readouterr().out
    assert "cmd_err= 5.0cm" in out


def test_render_monitor_rejects_missing_required_command_target():
    monitor = _load_monitor()
    st = _state()
    st["arms"]["left"]["cmd_pos"] = None
    monitor._validate_bimanual_state(st, require_hand_render=True)
    with pytest.raises(ValueError, match="arms.left.cmd_pos"):
        monitor._validate_bimanual_state(st, require_hand_render=True, require_command_target=True)


def test_render_monitor_bimanual_state_validation_rejects_missing_side():
    monitor = _load_monitor()
    st = _state()
    del st["op"]["hands"]["right"]
    with pytest.raises(ValueError, match="op.hands.right"):
        monitor._validate_bimanual_state(st, require_hand_render=True)


def test_render_monitor_rejects_schema_version_mismatch():
    monitor = _load_monitor()
    st = _state()
    st["v"] = topics.SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema version"):
        monitor._validate_bimanual_state(st, require_hand_render=True)


def test_render_monitor_accepts_body_relative_gated_untracked_hand():
    monitor = _load_monitor()
    st = _state()
    st["status"]["tracked"]["left"] = False
    st["op"]["hands"]["left"]["tracked"] = False
    st["op"]["hands"]["left"]["wrist_body"] = None
    monitor._validate_bimanual_state(st, require_hand_render=True)


def test_render_monitor_accepts_missing_headset_operator_payload():
    monitor = _load_monitor()
    st = _state()
    st["op"]["head_pos"] = None
    st["op"]["torso_pos"] = None
    for side in SIDES:
        st["status"]["tracked"][side] = False
        st["op"]["hands"][side]["tracked"] = False
        st["op"]["hands"][side]["wrist_body"] = None
    monitor._validate_bimanual_state(st, require_hand_render=True)


def test_render_monitor_rejects_malformed_operator_pose_vectors():
    monitor = _load_monitor()
    st = _state()
    st["op"]["torso_from_head"] = [0.0, -0.35]
    with pytest.raises(ValueError, match="op.torso_from_head"):
        monitor._validate_bimanual_state(st, require_hand_render=True)

    st = _state()
    st["op"]["head_pos"] = [0.0, 1.6, 0.0]
    st["op"]["torso_pos"] = None
    with pytest.raises(ValueError, match="head_pos and op.torso_pos"):
        monitor._validate_bimanual_state(st, require_hand_render=True)


def test_render_monitor_rejects_non_finite_arm_payloads():
    monitor = _load_monitor()
    st = _state()
    st["arms"]["right"]["link_pos"][0] = float("nan")
    with pytest.raises(ValueError, match="arms.right.link_pos.*finite"):
        monitor._validate_bimanual_state(st, require_hand_render=True)

    st = _state()
    st["arms"]["right"]["cmd_pos"] = [0.0, float("inf"), 0.0]
    with pytest.raises(ValueError, match="arms.right.cmd_pos.*finite"):
        monitor._validate_bimanual_state(st, require_hand_render=True)

    st = _state()
    st["arms"]["left"]["cmd_quat"] = [1.0, 0.0, float("inf"), 0.0]
    with pytest.raises(ValueError, match="arms.left.cmd_quat.*finite"):
        monitor._validate_bimanual_state(st, require_hand_render=True)


def test_render_monitor_rejects_non_finite_hand_and_operator_payloads():
    monitor = _load_monitor()
    st = _state()
    st["hand_render"]["left"]["q"][0] = float("nan")
    with pytest.raises(ValueError, match="hand_render.left.q.*finite"):
        monitor._validate_bimanual_state(st, require_hand_render=True)

    st = _state()
    st["op"]["hands"]["right"]["wrist_body"][1] = float("inf")
    with pytest.raises(ValueError, match="op.hands.right.wrist_body.*finite"):
        monitor._validate_bimanual_state(st, require_hand_render=True)


def test_render_monitor_rejects_status_operator_tracking_mismatch():
    monitor = _load_monitor()
    st = _state()
    st["status"]["tracked"]["left"] = False
    with pytest.raises(ValueError, match="status.tracked.left"):
        monitor._validate_bimanual_state(st, require_hand_render=True)


def test_render_monitor_rejects_untracked_wrist_body_vector():
    monitor = _load_monitor()
    st = _state()
    st["status"]["tracked"]["left"] = False
    st["op"]["hands"]["left"]["tracked"] = False
    with pytest.raises(ValueError, match="wrist_body must be null"):
        monitor._validate_bimanual_state(st, require_hand_render=True)


def test_render_monitor_hand_render_requires_orca_joint_order():
    monitor = _load_monitor()
    st = _state()
    st["hand_render"]["left"]["names"] = list(reversed(ORCA_JOINT_ORDER))
    with pytest.raises(ValueError, match="ORCA_JOINT_ORDER"):
        monitor._validate_bimanual_state(st, require_hand_render=True)


def test_render_monitor_require_frame_rejects_empty_stream():
    monitor = _load_monitor()
    with pytest.raises(RuntimeError, match="no render.state frames"):
        monitor._ensure_observed(0, require_frame=True)
    monitor._ensure_observed(0, require_frame=False)
    monitor._ensure_observed(1, require_frame=True)
