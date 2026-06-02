"""Compose the real bimanual rig in MuJoCo: AgileX stand + two 5-DoF YAM arms
(the friend's CAD-measured model) + two ORCA hands on the flanges.

Each arm is a standalone sub-model (yam_{side}.mjcf assets+actuators + the
arm-only body with a "{side}_ee" flange site), attached into the world at its
CAD base pose; the ORCA hand is then attached at the flange. The SAME arm
sub-model XML (arm_xml) is reused by arms/ik.py as the standalone IK model, so
sim and IK share one source of truth.

The ORCA hand models are referenced from the installed orca_sim package; the YAM
arm + AgileX stand are vendored under sim/models/yam_real/.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import orca_sim

from ..config import SIDES
from ..hands.joint_map import ActuatorRef, parse_actuators

YAM_DIR = Path(__file__).parent / "models" / "yam_real"
STAND = YAM_DIR / "assets" / "stand"
ARM_JOINTS = [f"arm_j{i}" for i in range(1, 6)]   # 5-DoF


def _orca_model_dir() -> Path:
    return Path(orca_sim.__file__).parent / "models" / "v2"


def arm_xml(side: str) -> str:
    """Standalone single-arm model: YAM assets+actuators + arm-only body + flange
    site. Used both as an IK model (arms/ik.py) and as the sim sub-model to attach."""
    mjcf = YAM_DIR / "mjcf"
    return f"""<mujoco model="yam_{side}_arm">
  <compiler angle="radian"/>
  <default>
    <joint damping="2.0" armature="0.1"/>
    <geom type="mesh" contype="0" conaffinity="0" group="2" rgba="0.75 0.75 0.78 1"/>
  </default>
  <include file="{mjcf / f'yam_{side}.mjcf'}"/>
  <worldbody>
    <include file="{mjcf / f'yam_{side}_body.xml'}"/>
  </worldbody>
</mujoco>"""


def _hand_spec(side: str) -> mujoco.MjSpec:
    md = _orca_model_dir()
    opts = md / "assets" / "options.xml"
    mjcf = md / "mjcf"
    xml = f"""<mujoco model="orcahand_{side}_nofloor">
  <include file="{opts}"/>
  <include file="{mjcf / f'orcahand_{side}.mjcf'}"/>
  <worldbody>
    <body name="{side}_hand_base"><include file="{mjcf / f'orcahand_{side}_body.xml'}"/></body>
  </worldbody>
</mujoco>"""
    return mujoco.MjSpec.from_string(xml)


@dataclass
class SimInfo:
    model: mujoco.MjModel
    actuators: list[ActuatorRef]
    arm_act: dict[str, list[int]] = field(default_factory=dict)
    hand_act: dict[str, list[tuple[int, str]]] = field(default_factory=dict)
    ee_site: dict[str, int] = field(default_factory=dict)

    def arm_qadr(self, side: str) -> list[int]:
        m = self.model
        return [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_arm_j{i}")])
                for i in range(1, 6)]


def _base_scene_xml() -> str:
    """World shell: floor, lights, and the AgileX stand (visual-only, feet on z=0)."""
    return f"""<mujoco model="bimanual_yam_orca">
  <compiler angle="radian"/>
  <option gravity="0 0 0" integrator="implicitfast"/>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <asset>
    <mesh name="stand_part0" file="{STAND / 'frame_part0.stl'}" scale="0.001 0.001 0.001"/>
    <mesh name="stand_part1" file="{STAND / 'frame_part1.stl'}" scale="0.001 0.001 0.001"/>
  </asset>
  <worldbody>
    <light pos="0.4 -0.4 2.5" dir="-0.2 0.2 -1" directional="true"/>
    <light pos="-0.6 0.6 2.0" dir="0.3 -0.3 -1"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.3 0.3 0.35 1" contype="0" conaffinity="0"/>
    <body name="stand" pos="0 0 0.2475">
      <geom type="mesh" mesh="stand_part0" contype="0" conaffinity="0" rgba="0.62 0.64 0.68 1"/>
      <geom type="mesh" mesh="stand_part1" contype="0" conaffinity="0" rgba="0.62 0.64 0.68 1"/>
    </body>
  </worldbody>
</mujoco>"""


def build_model(rig: dict) -> SimInfo:
    spec = mujoco.MjSpec.from_string(_base_scene_xml())
    for side in SIDES:
        a = rig["arms"][side]
        base = spec.worldbody.add_frame(name=f"{side}_base", pos=a["base_pos"], quat=a["base_quat"])
        spec.attach(mujoco.MjSpec.from_string(arm_xml(side)), prefix="", frame=base)
        link5 = spec.body(f"{side}_arm_link5")
        flange = link5.add_frame(name=f"{side}_flange", pos=a["hand_pos"], euler=a["hand_euler"])
        # Hands get a per-side prefix so the two ORCA models' shared materials/
        # defaults (white/black/... from options.xml) don't collide. Arms stay
        # unprefixed (rgba only, no clash) → clean joint names left_arm_j1 etc.
        spec.attach(_hand_spec(side), prefix=f"{side[0]}h_", frame=flange)

    model = spec.compile()
    refs = parse_actuators(model)
    info = SimInfo(model=model, actuators=refs)
    for side in SIDES:
        arm = {r.target: r.id for r in refs if r.kind == "arm" and r.side == side}
        info.arm_act[side] = [arm[f"arm_j{i}"] for i in range(1, 6)]
        info.hand_act[side] = [(r.id, r.target) for r in refs if r.kind == "hand" and r.side == side]
        info.ee_site[side] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee")
    return info


def set_neutral(info: SimInfo, data: mujoco.MjData, rig: dict, hand_neutral: dict[str, dict]) -> None:
    import math
    m = info.model
    for side in SIDES:
        q = rig["arms"][side]["neutral_q"]
        for adr, val, aid in zip(info.arm_qadr(side), q, info.arm_act[side]):
            data.qpos[adr] = val
            data.ctrl[aid] = val
        for aid, joint in info.hand_act[side]:
            val = math.radians(hand_neutral[side].get(joint, 0.0))
            jid = m.actuator_trnid[aid, 0]
            data.qpos[m.jnt_qposadr[jid]] = val
            data.ctrl[aid] = val
    mujoco.mj_forward(m, data)
