"""Per-arm differential IK for the 6-DoF YAM, using mink on a STANDALONE single-arm
model (so one arm's IK can never perturb the other). Poses are in the arm's base
frame; the arm XML is shared with the sim composer (sim.model.arm_xml).

TWO-STAGE solve (the key to clean teleop on this arm):
  1. POSITION — move j1-j3 (the arm) to the target wrist position. j4-j6 frozen.
  2. ORIENTATION — move j4-j6 (the wrist) to match the target hand orientation.
     j1-j3 frozen.
So the arm never swings to orient the wrist (no jerk), and the hand orientation is
matched about the correct axes by real IK (no "tilted axis" from a hand-rolled
pitch/yaw/roll decomposition). 6 DoF = position + full orientation achievable.
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

        # SOFT joint limits = home ± margin (clamped to hard limits) so the IK
        # physically cannot fold the arm into a buckle.
        hard_lo = np.asarray(rig["arms"]["joint_limits"]["lower"], dtype=float)
        hard_hi = np.asarray(rig["arms"]["joint_limits"]["upper"], dtype=float)
        margin = np.asarray(ik.get("soft_margin", [1.4, 1.0, 1.4, 1.4, 1.5, 1.7]), dtype=float)
        self.soft_lo = np.maximum(self.q0 - margin, hard_lo)
        self.soft_hi = np.minimum(self.q0 + margin, hard_hi)
        for i, j in enumerate(self.joints):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            self.model.jnt_range[jid] = [self.soft_lo[i], self.soft_hi[i]]

        self.config = mink.Configuration(self.model)
        # POSITION targets the WRIST site (before the wrist joints → invariant to
        # j4/j5/j6); ORIENTATION targets the hand (ee site).
        self.pos_task = mink.FrameTask(frame_name=f"{side}_wrist", frame_type="site",
                                       position_cost=ik["pos_cost"], orientation_cost=0.0,
                                       lm_damping=ik["lm_damping"])
        self.ori_task = mink.FrameTask(frame_name=f"{side}_ee", frame_type="site",
                                       position_cost=0.0, orientation_cost=ik["ori_cost"],
                                       lm_damping=ik["lm_damping"])
        self.posture = mink.PostureTask(self.model, cost=ik["posture_cost"])

        arm, wrist = self.joints[:3], self.joints[3:]
        mv = ik["max_vel"]
        froze = 1e-4
        clim = mink.ConfigurationLimit(self.model)
        # Stage 1: arm joints free, wrist frozen.  Stage 2: wrist free, arm frozen.
        self.limits_pos = [clim, mink.VelocityLimit(
            self.model, {**{j: mv for j in arm}, **{j: froze for j in wrist}})]
        self.limits_ori = [clim, mink.VelocityLimit(
            self.model, {**{j: froze for j in arm}, **{j: mv for j in wrist}})]
        self.solver = ik.get("solver", "daqp")
        self.damping = float(ik.get("damping", 1e-3))
        self.dt = 1.0 / rig["control"]["arm_hz"]
        self.iters = int(ik.get("iters", 4))
        self.reset()

    def reset(self) -> None:
        self.config.update(self.q0)
        self.posture.set_target(self.q0)

    def seed(self, q: np.ndarray) -> None:
        self.config.update(np.asarray(q, dtype=float))

    @property
    def q(self) -> np.ndarray:
        return self.config.q.copy()

    def fk_ee(self) -> mink.SE3:
        return self.config.get_transform_frame_to_world(f"{self.side}_ee", "site")

    def fk_wrist(self) -> mink.SE3:
        return self.config.get_transform_frame_to_world(f"{self.side}_wrist", "site")

    def solve(self, target: mink.SE3, iters: int | None = None) -> np.ndarray:
        iters = self.iters if iters is None else iters
        self.pos_task.set_target(target)
        self.ori_task.set_target(target)
        for _ in range(iters):   # stage 1: position → arm joints j1-j3
            v = mink.solve_ik(self.config, [self.pos_task, self.posture], self.dt,
                              self.solver, damping=self.damping, limits=self.limits_pos)
            self.config.integrate_inplace(v, self.dt)
        for _ in range(iters):   # stage 2: orientation → wrist joints j4-j6
            v = mink.solve_ik(self.config, [self.ori_task], self.dt,
                              self.solver, damping=self.damping, limits=self.limits_ori)
            self.config.integrate_inplace(v, self.dt)
        return self.config.q.copy()
