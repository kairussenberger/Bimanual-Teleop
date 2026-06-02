"""Name maps between ORCA-core joints (degrees) and the MuJoCo sim model.

The sim model is built by sim/model.py: each YAM arm is attached with prefix
``l_``/``r_`` and each ORCA hand with the same prefix, so a left index-MCP
actuator is e.g. ``l_left_i-mcp_actuator`` and a left arm joint is ``l_joint1``.

ORCA-core joint ids:  wrist, thumb_{cmc,abd,mcp,dip}, {index,middle,ring,pinky}_{abd,mcp,pip}
Sim short names:      wrist, t-{cmc,abd,mcp,pip},     {i,m,r,p}-{abd,mcp,pip}
(note ORCA's ``thumb_dip`` is the sim's ``t-pip`` — same physical joint.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco

_FINGER_SHORT = {"index": "i", "middle": "m", "ring": "r", "pinky": "p", "thumb": "t"}
_SHORT_FINGER = {v: k for k, v in _FINGER_SHORT.items()}


def orca_to_sim_short(orca_joint: str) -> str:
    """'index_mcp' -> 'i-mcp'; 'thumb_dip' -> 't-pip'; 'wrist' -> 'wrist'."""
    if orca_joint == "wrist":
        return "wrist"
    finger, _, joint = orca_joint.partition("_")
    if finger == "thumb" and joint == "dip":
        joint = "pip"
    return f"{_FINGER_SHORT[finger]}-{joint}"


def sim_short_to_orca(short: str) -> str:
    """Inverse of orca_to_sim_short. 't-pip' -> 'thumb_dip'."""
    if short == "wrist":
        return "wrist"
    f, _, j = short.partition("-")
    finger = _SHORT_FINGER[f]
    if finger == "thumb" and j == "pip":
        j = "dip"
    return f"{finger}_{j}"


@dataclass
class ActuatorRef:
    id: int
    name: str
    kind: str   # 'arm' or 'hand'
    side: str   # 'left' or 'right'
    target: str  # arm: 'joint1'..; hand: orca joint id ('index_mcp', 'wrist', ...)


def parse_actuators(model: mujoco.MjModel) -> list[ActuatorRef]:
    """Classify every actuator in the combined model by side/kind/target."""
    refs: list[ActuatorRef] = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if not name or name[1] != "_":
            raise ValueError(f"unexpected actuator name (no l_/r_ prefix): {name!r}")
        side = "left" if name[0] == "l" else "right"
        rest = name[2:]
        if rest.startswith("joint") and rest.endswith("_act"):
            refs.append(ActuatorRef(i, name, "arm", side, rest[: -len("_act")]))
            continue
        core = rest.replace("left_", "").replace("right_", "").replace("_actuator", "")
        refs.append(ActuatorRef(i, name, "hand", side, sim_short_to_orca(core)))
    return refs


def load_hand_config(model_name: str):
    """neutral_position + joint_roms (both dicts in degrees) for an ORCA hand."""
    from orca_core.hand_config import OrcaHandConfig
    cfg = OrcaHandConfig.from_config_path(model_name=model_name)
    return dict(cfg.neutral_position), dict(cfg.joint_roms_dict)


def deg(value: float) -> float:
    return math.radians(value)
