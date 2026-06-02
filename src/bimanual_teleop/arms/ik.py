"""Per-arm differential IK for a 5-DoF YAM, using mink on a STANDALONE single-arm
model (so left-arm IK can never perturb the right arm). All poses are in the
arm's own base frame. The arm model XML is shared with the sim composer
(sim.model.arm_xml) so IK and sim use one source of truth.

5-DoF note: a 5-joint arm cannot reach an arbitrary 6-DoF end-effector pose. mink
solves it as a weighted least-squares — we weight POSITION above orientation
(rig ik.pos_cost > ik.ori_cost) so the flange tracks position well and orientation
is best-effort, which matches a wrist whose roll is fixed by the hand mount.
"""
from __future__ import annotations

import mink
import mujoco
import numpy as np


class ArmIK:
    def __init__(self, rig: dict, side: str):
        from ..sim.model import arm_xml  # shared model source
        self.side = side
        self.joints = [f"{side}_arm_j{i}" for i in range(1, 6)]
        self.model = mujoco.MjModel.from_xml_string(arm_xml(side))
        self.config = mink.Configuration(self.model)
        ik = rig["ik"]
        self.ee_task = mink.FrameTask(
            frame_name=f"{side}_ee", frame_type="site",
            position_cost=ik["pos_cost"], orientation_cost=ik["ori_cost"],
            lm_damping=ik["lm_damping"],
        )
        self.posture = mink.PostureTask(self.model, cost=ik["posture_cost"])
        self.tasks = [self.ee_task, self.posture]
        self.limits = [
            mink.ConfigurationLimit(self.model),
            mink.VelocityLimit(self.model, {j: ik["max_vel"] for j in self.joints}),
        ]
        self.solver = ik.get("solver", "daqp")
        self.damping = float(ik.get("damping", 1e-3))
        self.dt = 1.0 / rig["control"]["arm_hz"]
        self.q0 = np.asarray(rig["arms"][side]["neutral_q"], dtype=float)
        self.reset()

    def reset(self) -> None:
        self.config.update(self.q0)
        self.posture.set_target(self.q0)

    def seed(self, q: np.ndarray) -> None:
        """Sync the IK config to a measured/known joint state (e.g. on hardware
        engage, from YamArm.state()) so anchoring matches the real arm."""
        self.config.update(np.asarray(q, dtype=float))

    @property
    def q(self) -> np.ndarray:
        return self.config.q.copy()

    def fk_ee(self) -> mink.SE3:
        return self.config.get_transform_frame_to_world(f"{self.side}_ee", "site")

    def solve(self, target: mink.SE3, iters: int = 1) -> np.ndarray:
        self.ee_task.set_target(target)
        for _ in range(iters):
            vel = mink.solve_ik(self.config, self.tasks, self.dt, self.solver,
                                damping=self.damping, limits=self.limits)
            self.config.integrate_inplace(vel, self.dt)
        return self.config.q.copy()
