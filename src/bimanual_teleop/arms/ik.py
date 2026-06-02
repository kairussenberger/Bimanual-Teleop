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
        self.joints = [f"{side}_arm_j{i}" for i in range(1, 7)]   # 6-DoF
        self.model = mujoco.MjModel.from_xml_string(arm_xml(side))
        ik = rig["ik"]
        self.q0 = np.asarray(rig["arms"][side]["neutral_q"], dtype=float)

        # SOFT joint limits = home ± margin (clamped to the URDF hard limits),
        # applied to the model BEFORE building the limit so the IK physically
        # cannot drive a joint past them — the arm can never fold into a buckle.
        # Tighter ROM is an accepted trade for never buckling.
        hard_lo = np.asarray(rig["arms"]["joint_limits"]["lower"], dtype=float)
        hard_hi = np.asarray(rig["arms"]["joint_limits"]["upper"], dtype=float)
        margin = np.asarray(ik.get("soft_margin", [1.4, 1.2, 1.4, 1.4, 1.5]), dtype=float)
        self.soft_lo = np.maximum(self.q0 - margin, hard_lo)
        self.soft_hi = np.minimum(self.q0 + margin, hard_hi)
        for i, j in enumerate(self.joints):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            self.model.jnt_range[jid] = [self.soft_lo[i], self.soft_hi[i]]

        self.config = mink.Configuration(self.model)
        self.ee_task = mink.FrameTask(
            frame_name=f"{side}_ee", frame_type="site",
            position_cost=ik["pos_cost"], orientation_cost=ik["ori_cost"],
            lm_damping=ik["lm_damping"],
        )
        self.posture = mink.PostureTask(self.model, cost=ik["posture_cost"])
        self.tasks = [self.ee_task, self.posture]
        # Position IK uses j1-j3; the wrist joints j4-j6 are velocity-frozen here
        # and set directly from the operator's wrist (so orientation never recruits
        # the arm joints → no jerk).
        wmv = float(ik.get("wrist_max_vel", 0.02))
        vlim = {j: ik["max_vel"] for j in self.joints[:3]}
        vlim.update({j: wmv for j in self.joints[3:]})
        self.limits = [
            mink.ConfigurationLimit(self.model),       # now reads the soft ranges
            mink.VelocityLimit(self.model, vlim),
        ]
        self.solver = ik.get("solver", "daqp")
        self.damping = float(ik.get("damping", 1e-3))
        self.dt = 1.0 / rig["control"]["arm_hz"]
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

    def set_wrist(self, j456: np.ndarray) -> None:
        """Set the wrist joints (j4,j5,j6) directly, clamped to their soft range."""
        q = self.config.q.copy()
        q[3:6] = np.clip(np.asarray(j456, dtype=float), self.soft_lo[3:6], self.soft_hi[3:6])
        self.config.update(q)

    def solve(self, target: mink.SE3, iters: int = 1) -> np.ndarray:
        self.ee_task.set_target(target)
        for _ in range(iters):
            vel = mink.solve_ik(self.config, self.tasks, self.dt, self.solver,
                                damping=self.damping, limits=self.limits)
            self.config.integrate_inplace(vel, self.dt)
        return self.config.q.copy()
