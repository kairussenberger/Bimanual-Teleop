"""Compose the bimanual MuJoCo model: torso + 2 YAM arms + 2 ORCA hands.

Strategy (validated against mujoco 3.9 mjSpec): each sub-model is loaded as its
own spec — so its own meshdir/defaults/assets resolve cleanly — then attached
into the world with a side prefix (``l_``/``r_``). The ORCA hands are referenced
from the installed ``orca_sim`` package (no 200 MB of meshes vendored here); the
YAM arm is vendored under sim/models/yam/ (with an EE site + position actuators
added). Whole-spec ``attach`` carries each hand's 17 actuators + contact excludes.

IK does NOT use this combined model — it uses a standalone single-arm YAM model
(see arms/ik.py) so left-arm IK can't perturb the right arm. This model is for
simulation/visualization and for applying joint/finger commands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import orca_sim

from ..config import SIDES
from ..hands.joint_map import ActuatorRef, parse_actuators

YAM_XML = Path(__file__).parent / "models" / "yam" / "yam.xml"


def _orca_model_dir() -> Path:
    return Path(orca_sim.__file__).parent / "models" / "v2"


def _hand_spec(side: str) -> mujoco.MjSpec:
    """A floor-less complete ORCA hand model (options + mjcf + body), referencing
    the installed orca_sim assets via absolute include paths (portable)."""
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
    """Indices/metadata the controllers and sim loop need."""
    model: mujoco.MjModel
    actuators: list[ActuatorRef]
    # per side -> arm: list of 6 actuator ids in joint1..6 order
    arm_act: dict[str, list[int]] = field(default_factory=dict)
    # per side -> ordered list of (actuator_id, orca_joint) for the 17 hand dofs
    hand_act: dict[str, list[tuple[int, str]]] = field(default_factory=dict)
    ee_site: dict[str, int] = field(default_factory=dict)   # per side -> site id of l_ee/r_ee

    def arm_qadr(self, side: str) -> list[int]:
        m = self.model
        adr = []
        for j in range(1, 7):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side[0]}_joint{j}")
            adr.append(int(m.jnt_qposadr[jid]))
        return adr


def build_model(rig: dict) -> SimInfo:
    spec = mujoco.MjSpec()
    spec.option.gravity = [0.0, 0.0, 0.0]      # teleop preview: clean IK tracking, no sag
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.visual.global_.offwidth = 1280        # offscreen framebuffer (tools/snapshots)
    spec.visual.global_.offheight = 960

    wb = spec.worldbody
    wb.add_light(pos=[0.4, -0.4, 2.5], dir=[-0.2, 0.2, -1.0],
                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    wb.add_light(pos=[-0.6, 0.6, 2.0], dir=[0.3, -0.3, -1.0])
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1],
                rgba=[0.3, 0.3, 0.35, 1], contype=0, conaffinity=0)

    t = rig["torso"]
    torso = wb.add_body(name="torso", pos=t["pos"])
    torso.add_geom(name="torso_box", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[t["half_width"], t["half_depth"], t["half_height"]],
                   rgba=[0.55, 0.57, 0.6, 1], contype=0, conaffinity=0)
    # a short post down to the floor, just for visual grounding
    torso.add_geom(name="torso_post", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   fromto=[0, 0, -t["half_height"], 0, 0, -t["pos"][2]],
                   size=[0.03], rgba=[0.4, 0.4, 0.45, 1], contype=0, conaffinity=0)

    for side in SIDES:
        a = rig["arms"][side]
        p = side[0]  # 'l' / 'r'
        shoulder = torso.add_frame(name=f"{side}_shoulder", pos=a["mount_pos"], euler=a["mount_euler"])
        spec.attach(mujoco.MjSpec.from_file(str(YAM_XML)), prefix=f"{p}_", frame=shoulder)
        link6 = spec.body(f"{p}_link6")
        wrist = link6.add_frame(name=f"{side}_wrist_mount", pos=a["hand_pos"], euler=a["hand_euler"])
        spec.attach(_hand_spec(side), prefix=f"{p}_", frame=wrist)

    model = spec.compile()

    refs = parse_actuators(model)
    info = SimInfo(model=model, actuators=refs)
    for side in SIDES:
        arm = {r.target: r.id for r in refs if r.kind == "arm" and r.side == side}
        info.arm_act[side] = [arm[f"joint{j}"] for j in range(1, 7)]
        info.hand_act[side] = [(r.id, r.target) for r in refs if r.kind == "hand" and r.side == side]
        info.ee_site[side] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side[0]}_ee")
    return info


def set_neutral(info: SimInfo, data: mujoco.MjData, rig: dict,
                hand_neutral: dict[str, dict]) -> None:
    """Place arms at neutral_q and hands at their config neutral (degrees→rad),
    and seed ctrl so position actuators hold neutral."""
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
