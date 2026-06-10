"""Tiny monitor for the `render.state` stream.

It can read either the ZMQ/msgpack topic or the Unity TCP newline-JSON stream, so
you can confirm the Python half end-to-end with no headset and no Unity Editor.

    # terminal 1
    uv run python -m bimanual_teleop.launch.run_teleop --vr fake
    # terminal 2
    uv run python scripts/render_monitor.py
    uv run python scripts/render_monitor.py --transport json
"""
from __future__ import annotations

import argparse
import json
import socket
import time

import numpy as np

from bimanual_teleop.bus import topics
from bimanual_teleop.bus.zmq_io import LatestSub
from bimanual_teleop.hands.joint_map import ORCA_JOINT_ORDER


def _json_host_port(endpoint: str) -> tuple[str, int]:
    host, port_s = endpoint.removeprefix("tcp://").rsplit(":", 1)
    return host, int(port_s)


def _hand_summary(st: dict, side: str, required: bool) -> str:
    hr = st.get("hand_render", {}).get(side)
    if hr is None:
        if required:
            raise ValueError(f"missing hand_render.{side}")
        return "hand=missing"
    names = hr.get("names") or []
    q = hr.get("q") or []
    if len(names) != len(q):
        raise ValueError(f"hand_render.{side} names/q length mismatch: {len(names)} != {len(q)}")
    if required and names != ORCA_JOINT_ORDER:
        raise ValueError(f"hand_render.{side}.names must match ORCA_JOINT_ORDER")
    q_arr = _require_vec(f"hand_render.{side}.q", q, len(q))
    vals = {n: float(v) for n, v in zip(names, q_arr)}
    curl_keys = [k for k in ("index_mcp", "index_pip", "middle_mcp", "middle_pip",
                             "ring_mcp", "ring_pip", "pinky_mcp", "pinky_pip")
                 if k in vals]
    curl = float(np.mean([vals[k] for k in curl_keys])) if curl_keys else 0.0
    return f"hand17={len(q):02d} curl={curl:5.1f}deg"


def _require_vec(name: str, value, n: int) -> np.ndarray:
    try:
        ok = value is not None and len(value) == n
    except TypeError:
        ok = False
    if not ok:
        raise ValueError(f"{name} must have length {n}")
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must contain numeric values") from e
    if arr.shape != (n,):
        raise ValueError(f"{name} must have length {n}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr


def _require_nullable_vec(name: str, value, n: int) -> None:
    if value is None:
        return
    _require_vec(name, value, n)


def _validate_bimanual_state(st: dict, *, require_hand_render: bool = False,
                             require_command_target: bool = False) -> None:
    if st.get("v") != topics.SCHEMA_VERSION:
        raise ValueError(f"render.state schema version must be {topics.SCHEMA_VERSION}")

    op = st.get("op", {})
    _require_vec("op.torso_from_head", op.get("torso_from_head"), 3)
    _require_nullable_vec("op.head_pos", op.get("head_pos"), 3)
    _require_nullable_vec("op.torso_pos", op.get("torso_pos"), 3)
    if (op.get("head_pos") is None) != (op.get("torso_pos") is None):
        raise ValueError("op.head_pos and op.torso_pos must both be null or both be vectors")

    for side in ("left", "right"):
        arm = st.get("arms", {}).get(side)
        if arm is None:
            raise ValueError(f"missing arms.{side}")
        _require_vec(f"arms.{side}.q", arm.get("q"), 6)
        _require_vec(f"arms.{side}.link_pos", arm.get("link_pos"), 24)
        _require_vec(f"arms.{side}.ee_pos", arm.get("ee_pos"), 3)
        _require_vec(f"arms.{side}.ee_quat", arm.get("ee_quat"), 4)
        if require_command_target:
            _require_vec(f"arms.{side}.cmd_pos", arm.get("cmd_pos"), 3)
        else:
            _require_nullable_vec(f"arms.{side}.cmd_pos", arm.get("cmd_pos"), 3)
        _require_nullable_vec(f"arms.{side}.cmd_quat", arm.get("cmd_quat"), 4)
        _require_nullable_vec(f"arms.{side}.margins", arm.get("margins"), 6)
        if require_hand_render:
            _hand_summary(st, side, required=True)

        status = st.get("status", {})
        if side not in status.get("engaged", {}) or side not in status.get("tracked", {}):
            raise ValueError(f"missing status flags for {side}")

        op_hand = op.get("hands", {}).get(side)
        if op_hand is None:
            raise ValueError(f"missing op.hands.{side}")
        status_tracked = bool(status.get("tracked", {}).get(side, False))
        op_tracked = bool(op_hand.get("tracked"))
        if op_tracked != status_tracked:
            raise ValueError(f"status.tracked.{side} must match op.hands.{side}.tracked")
        if op_tracked:
            _require_vec(f"op.hands.{side}.wrist_body", op_hand.get("wrist_body"), 3)
        elif op_hand.get("wrist_body") is not None:
            raise ValueError(f"op.hands.{side}.wrist_body must be null when untracked")
        _require_nullable_vec(f"op.hands.{side}.raw_wrist", op_hand.get("raw_wrist"), 3)

    if "hz" in st.get("status", {}):
        _require_vec("status.hz", [st["status"]["hz"]], 1)


def _ensure_observed(count: int, require_frame: bool) -> None:
    if require_frame and count <= 0:
        raise RuntimeError("no render.state frames were observed")


def _print_state(st: dict, *, require_hand_render: bool = False,
                 require_bimanual_state: bool = False,
                 require_command_target: bool = False) -> None:
    if require_bimanual_state:
        _validate_bimanual_state(
            st,
            require_hand_render=require_hand_render,
            require_command_target=require_command_target,
        )
    a = st["arms"]["right"]
    q = np.round(a["q"], 3)
    ee = np.round(a["ee_pos"], 3)
    cmd = a.get("cmd_pos")
    cmd_err = "cmd_err=none"
    if cmd is not None:
        cmd_err = f"cmd_err={np.linalg.norm(np.asarray(cmd, float) - np.asarray(a['ee_pos'], float)) * 100.0:4.1f}cm"
    status = st["status"]
    eng = "".join(s[0].upper() if status["engaged"][s] else "." for s in ("left", "right"))
    trk = "".join(s[0].upper() if status["tracked"][s] else "." for s in ("left", "right"))
    op = st.get("op", {}).get("hands", {}).get("right", {})
    wb = op.get("wrist_body")
    wb_s = "none" if wb is None else np.array2string(np.round(np.asarray(wb, float), 3), separator=",")
    hand_s = _hand_summary(st, "right", require_hand_render)
    print(f"hz={status['hz']:5.1f} eng={eng} trk={trk} | R q={q} ee={ee} {cmd_err} wrist_body={wb_s} {hand_s}", flush=True)


def _monitor_zmq(endpoint: str, seconds: float, require_hand_render: bool,
                 require_bimanual_state: bool, require_frame: bool,
                 require_command_target: bool) -> None:
    sub = LatestSub(endpoint, topics.RENDER_STATE)
    print(f"subscribing to {topics.RENDER_STATE} @ {endpoint} ... (Ctrl+C to stop)")
    t0 = time.monotonic()
    last = None
    seen = 0
    try:
        while True:
            sub.poll()
            st = sub.get(topics.RENDER_STATE)
            if st is not None and st.get("stamp") != last:
                last = st.get("stamp")
                _print_state(st, require_hand_render=require_hand_render,
                             require_bimanual_state=require_bimanual_state,
                             require_command_target=require_command_target)
                seen += 1
            if seconds and (time.monotonic() - t0) >= seconds:
                break
            time.sleep(0.05)
        _ensure_observed(seen, require_frame)
    finally:
        sub.close()


def _monitor_json(endpoint: str, seconds: float, require_hand_render: bool,
                  require_bimanual_state: bool, require_frame: bool,
                  require_command_target: bool) -> None:
    host, port = _json_host_port(endpoint)
    print(f"connecting to Unity JSON @ {host}:{port} ... (Ctrl+C to stop)")
    t0 = time.monotonic()
    seen = 0
    with socket.create_connection((host, port), timeout=3.0) as sock:
        sock.settimeout(0.25)
        f = sock.makefile("r")
        while True:
            try:
                line = f.readline()
            except socket.timeout:
                line = ""
            if line:
                _print_state(json.loads(line), require_hand_render=require_hand_render,
                             require_bimanual_state=require_bimanual_state,
                             require_command_target=require_command_target)
                seen += 1
            if seconds and (time.monotonic() - t0) >= seconds:
                break
    _ensure_observed(seen, require_frame)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transport", choices=["zmq", "json"], default="zmq")
    ap.add_argument("--endpoint", default=topics.RENDER_ENDPOINT)
    ap.add_argument("--json-endpoint", default="tcp://127.0.0.1:8102")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = until Ctrl+C")
    ap.add_argument("--require-hand-render", action="store_true",
                    help="fail if the Unity fixed-shape hand_render block is missing")
    ap.add_argument("--require-bimanual-state", action="store_true",
                    help="fail if arms/status/operator wrist_body are missing for either side")
    ap.add_argument("--require-command-target", action="store_true",
                    help="fail if either arm lacks a finite commanded EE target")
    ap.add_argument("--require-frame", action="store_true",
                    help="fail if no render.state frame is observed before --seconds elapses")
    args = ap.parse_args()

    try:
        if args.transport == "json":
            _monitor_json(args.json_endpoint, args.seconds, args.require_hand_render,
                          args.require_bimanual_state, args.require_frame,
                          args.require_command_target)
        else:
            _monitor_zmq(args.endpoint, args.seconds, args.require_hand_render,
                         args.require_bimanual_state, args.require_frame,
                         args.require_command_target)
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
