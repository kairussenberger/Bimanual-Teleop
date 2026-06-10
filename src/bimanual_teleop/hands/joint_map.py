"""Name maps between ORCA-core joints (degrees) and compact render/debug names.

These helpers are deliberately small and model-independent: ORCA-core exposes
human-readable joint ids, while telemetry/render/debug output often wants the
short names that were historically used by the old simulator.

ORCA-core joint ids:  wrist, thumb_{cmc,abd,mcp,dip}, {index,middle,ring,pinky}_{abd,mcp,pip}
Short names:          wrist, t-{cmc,abd,mcp,pip},     {i,m,r,p}-{abd,mcp,pip}
(note ORCA's ``thumb_dip`` maps to ``t-pip`` — same physical joint.)
"""
from __future__ import annotations

import math

_FINGER_SHORT = {"index": "i", "middle": "m", "ring": "r", "pinky": "p", "thumb": "t"}
_SHORT_FINGER = {v: k for k, v in _FINGER_SHORT.items()}

ORCA_JOINT_ORDER = [
    "wrist",
    "thumb_cmc", "thumb_abd", "thumb_mcp", "thumb_dip",
    "index_abd", "index_mcp", "index_pip",
    "middle_abd", "middle_mcp", "middle_pip",
    "ring_abd", "ring_mcp", "ring_pip",
    "pinky_abd", "pinky_mcp", "pinky_pip",
]


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


def load_hand_config(model_name: str):
    """neutral_position + joint_roms (both dicts in degrees) for an ORCA hand.

    Prefers the orca_core driver package when installed (it carries any
    machine-local calibration); otherwise uses the configs vendored alongside
    the hand model (scripts/vendor_orcahand.py), so the sim/render/replay path
    needs no driver install at all."""
    try:
        from orca_core.hand_config import OrcaHandConfig
        cfg = OrcaHandConfig.from_config_path(model_name=model_name)
        return dict(cfg.neutral_position), dict(cfg.joint_roms_dict)
    except ImportError:
        pass
    import pathlib

    import yaml
    side = "left" if "left" in model_name else "right"
    path = (pathlib.Path(__file__).resolve().parents[1] / "sim" / "models"
            / "orcahand_v2" / f"hand_config_{side}.yaml")
    data = yaml.safe_load(path.read_text())
    return (dict(data["neutral_position"]),
            {k: tuple(v) for k, v in data["joint_roms"].items()})


def deg(value: float) -> float:
    return math.radians(value)
