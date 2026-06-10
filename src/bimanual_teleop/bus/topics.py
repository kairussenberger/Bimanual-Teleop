"""Message schema for the multi-process bus (hardware-grade path).

Messages are plain dicts encoded with msgpack (numpy arrays → lists). Every
message carries a schema version and a monotonic source stamp so consumers can
gate on staleness. Topics are decoupled producer→consumer channels; live control
hops use latest-value semantics (see zmq_io.LatestSub).
"""
from __future__ import annotations

import numpy as np
import msgpack

SCHEMA_VERSION = 2

# topic names
VR_POSE = "vr.pose"        # {v, stamp, head[16], hands:{side:{tracked, wrist[16], landmarks[75], pinch}}}
ARM_CMD = "arm.cmd"        # {v, stamp, side, q[6]}    target joint pos (rad)
ARM_STATE = "arm.state"    # {v, stamp, side, q[6], ee[16]}
HAND_CMD = "hand.cmd"      # {v, stamp, side, joints_deg:{orca_joint: deg}}
SAFETY = "safety.state"    # {v, stamp, state, engaged:{side:bool}, estop}
HEARTBEAT = "heartbeat"    # {v, stamp, who}

# Render channel (PC → renderers). One latest-wins message per tick carrying
# everything Unity needs to draw the robot + the spec Section-6 HUD:
#   {v, stamp,
#    arms:  {side: {q[6], link_pos[24], ee_pos[3], ee_quat[4]wxyz,
#                   cmd_pos[3]|None, cmd_quat[4]|None, margins[6]}},
#    hands: {side: {orca_joint: deg}},                 # dynamic dict for Python tools
#    hand_render: {side: {names[17], q[17]}},           # fixed-shape for Unity JsonUtility
#    op:    {torso_from_head[3], head_pos[3]|None, torso_pos[3]|None,
#            hands:{side:{tracked, wrist_body[3]|None, raw_wrist[3]|None}}},
#    status:{engaged:{side:bool}, tracked:{side:bool}, calib:{...}|None, hz}}
# ee_pos/quat and cmd_pos/quat are in the robot WORLD frame (achieved vs commanded target).
RENDER_STATE = "render.state"
# Publisher binds here; the Quest reaches it via `adb reverse tcp:8101 tcp:8101`
# (same mechanism the ORBIT ingest uses), or a LAN IP over Wi-Fi.
RENDER_ENDPOINT = "tcp://127.0.0.1:8101"


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"cannot serialize {type(o)}")


def pack(obj: dict) -> bytes:
    return msgpack.packb(obj, default=_default, use_bin_type=True)


def unpack(buf: bytes) -> dict:
    return msgpack.unpackb(buf, raw=False)


def msg(**fields) -> dict:
    """Build a message dict with the schema version stamped in."""
    return {"v": SCHEMA_VERSION, **fields}
