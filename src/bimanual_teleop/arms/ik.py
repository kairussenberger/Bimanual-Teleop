"""Per-arm differential IK for a 6-DoF YAM, using mink on a STANDALONE single-arm
model (so left-arm IK can never perturb the right arm). All poses are expressed
in the arm's own base frame (base at the origin of yam.xml).

The combined sim model (sim/model.py) only applies the resulting joint targets.
"""
from __future__ import annotations

from pathlib import Path

import mink
import mujoco
import numpy as np

YAM_XML = Path(__file__).resolve().parent.parent / "sim" / "models" / "yam" / "yam.xml"
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]


class ArmIK:
    """Wraps a mink Configuration + FrameTask(ee) + PostureTask + joint/vel limits."""

    def __init__(self, rig: dict, side: str):
        self.side = side
        self.model = mujoco.MjModel.from_xml_path(str(YAM_XML))
        self.config = mink.Configuration(self.model)
        ik = rig["ik"]
        self.ee_task = mink.FrameTask(
            frame_name="ee", frame_type="site",
            position_cost=ik["pos_cost"], orientation_cost=ik["ori_cost"],
            lm_damping=ik["lm_damping"],
        )
        self.posture = mink.PostureTask(self.model, cost=ik["posture_cost"])
        self.tasks = [self.ee_task, self.posture]
        self.limits = [
            mink.ConfigurationLimit(self.model),
            mink.VelocityLimit(self.model, {j: ik["max_vel"] for j in ARM_JOINTS}),
        ]
        self.solver = ik.get("solver", "daqp")
        self.damping = float(ik.get("damping", 1e-3))
        self.dt = 1.0 / rig["control"]["arm_hz"]

        self.q0 = np.asarray(rig["arms"][side]["neutral_q"], dtype=float)
        self.reset()

    def reset(self) -> None:
        self.config.update(self.q0)
        self.posture.set_target(self.q0)

    @property
    def q(self) -> np.ndarray:
        return self.config.q.copy()

    def fk_ee(self) -> mink.SE3:
        """Current EE pose (SE3) in the arm base frame."""
        return self.config.get_transform_frame_to_world("ee", "site")

    def solve(self, target: mink.SE3, iters: int = 1) -> np.ndarray:
        """One (or a few) diff-IK step(s) toward `target`; returns joint targets q (6,)."""
        self.ee_task.set_target(target)
        for _ in range(iters):
            vel = mink.solve_ik(self.config, self.tasks, self.dt, self.solver,
                                damping=self.damping, limits=self.limits)
            self.config.integrate_inplace(vel, self.dt)
        return self.config.q.copy()
