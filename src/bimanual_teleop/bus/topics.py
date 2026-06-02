"""Message schema for the multi-process bus (hardware-grade path).

Messages are plain dicts encoded with msgpack (numpy arrays → lists). Every
message carries a schema version and a monotonic source stamp so consumers can
gate on staleness. Topics are decoupled producer→consumer channels; live control
hops use latest-value semantics (see zmq_io.LatestSub).
"""
from __future__ import annotations

import numpy as np
import msgpack

SCHEMA_VERSION = 1

# topic names
VR_POSE = "vr.pose"        # {v, stamp, head[16], hands:{side:{tracked, wrist[16], landmarks[75], pinch}}}
ARM_CMD = "arm.cmd"        # {v, stamp, side, q[6]}    target joint pos (rad)
ARM_STATE = "arm.state"    # {v, stamp, side, q[6], ee[16]}
HAND_CMD = "hand.cmd"      # {v, stamp, side, joints_deg:{orca_joint: deg}}
SAFETY = "safety.state"    # {v, stamp, state, engaged:{side:bool}, estop}
HEARTBEAT = "heartbeat"    # {v, stamp, who}


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
