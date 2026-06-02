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
    """Classify every actuator in the combined model by side/kind/target.

    Names: arm (unprefixed) = ``{side}_arm_j{n}_actuator``; hand (per-side prefix
    ``lh_``/``rh_`` to namespace shared materials) = ``{p}h_{side}_{simshort}_actuator``
    (e.g. ``lh_left_i-mcp_actuator``)."""
    refs: list[ActuatorRef] = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name and (name.startswith("left_arm") or name.startswith("right_arm")):
            side = "left" if name.startswith("left") else "right"
            target = name[len(side) + 1:].removesuffix("_actuator")        # 'arm_j1'
            refs.append(ActuatorRef(i, name, "arm", side, target))
        elif name and (name.startswith("lh_") or name.startswith("rh_")):
            side = "left" if name.startswith("lh_") else "right"
            core = name[3:].removeprefix(f"{side}_").removesuffix("_actuator")  # 'wrist'/'i-mcp'
            refs.append(ActuatorRef(i, name, "hand", side, sim_short_to_orca(core)))
        else:
            raise ValueError(f"unexpected actuator name: {name!r}")
    return refs


def load_hand_config(model_name: str):
    """neutral_position + joint_roms (both dicts in degrees) for an ORCA hand."""
    from orca_core.hand_config import OrcaHandConfig
    cfg = OrcaHandConfig.from_config_path(model_name=model_name)
    return dict(cfg.neutral_position), dict(cfg.joint_roms_dict)


def deg(value: float) -> float:
    return math.radians(value)
