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

JOINT LIMITS & COLLISIONS (spec invariant #5 + Section 8):
  - Hard limits are baked into the MuJoCo model (jnt_range) AND enforced in the IK
    via mink.ConfigurationLimit. On top, SOFT limits cap each joint to home ± margin
    so the arm physically cannot buckle or HYPEREXTEND THE ELBOW (j3) past a
    human-plausible envelope. See limit_margins() for the live per-joint margin.
  - Self-collision avoidance: this IK runs on a STANDALONE single-arm model, so it
    cannot see the other arm/torso — cross-arm collision is mitigated upstream by
    the anti-cross world-Y guard + workspace box in arm_control. A CollisionAvoidance
    hook (collision_pairs=) is wired for when a combined/collidable model is used;
    the vendored arm geoms are visual-only (contype=0), so it stays OFF by default.
"""
from __future__ import annotations

import mink
import mujoco
import numpy as np


class ArmIK:
    ELBOW = 2   # index of j3, the elbow joint (soft limit caps its hyperextension)

    def __init__(self, rig: dict, side: str, collision_pairs=None):
        from ..sim.model import arm_xml  # shared model source
        self.side = side
        self.joints = [f"{side}_arm_j{i}" for i in range(1, 7)]   # 6-DoF
        self.model = mujoco.MjModel.from_xml_string(arm_xml(side))
        ik = rig["ik"]
        self.q0 = np.asarray(rig["arms"][side]["neutral_q"], dtype=float)

        # SOFT joint limits = home ± margin (clamped to hard limits) so the IK
        # physically cannot fold the arm into a buckle.
        self.hard_lo = np.asarray(rig["arms"]["joint_limits"]["lower"], dtype=float)
        self.hard_hi = np.asarray(rig["arms"]["joint_limits"]["upper"], dtype=float)
        margin = np.asarray(ik.get("soft_margin", [1.4, 1.0, 1.4, 1.4, 1.5, 1.7]), dtype=float)
        self.soft_lo = np.maximum(self.q0 - margin, self.hard_lo)
        self.soft_hi = np.minimum(self.q0 + margin, self.hard_hi)
        # Human-plausible ELBOW (invariant: no overextension). j3 is straightest at
        # home and bends BOTH ways, but only +j3 is anatomical flexion; −j3 swings the
        # forearm outward into a dislocated-looking bend. Floor j3 so it can't go there.
        # Tunable: ik.elbow_min (rad). Raise toward q0 (~0.305) if it still looks wrong;
        # set negative/None to disable.
        emin = ik.get("elbow_min", None)
        if emin is not None:
            self.soft_lo[self.ELBOW] = max(self.soft_lo[self.ELBOW], float(emin))
        emax = ik.get("elbow_max", None)
        if emax is not None:
            self.soft_hi[self.ELBOW] = min(self.soft_hi[self.ELBOW], float(emax))
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
        # Optional self-collision avoidance. OFF by default (the standalone arm
        # model is visual-only); pass geom_pairs to enable on a collidable model,
        # and it joins BOTH solve stages. See the class docstring for why cross-arm
        # collision is handled upstream instead.
        if collision_pairs:
            ca = mink.CollisionAvoidanceLimit(self.model, collision_pairs)
            self.limits_pos.append(ca)
            self.limits_ori.append(ca)
        self.solver = ik.get("solver", "daqp")
        self.damping = float(ik.get("damping", 1e-3))
        self.dt = 1.0 / rig["control"]["arm_hz"]
        self.iters = int(ik.get("iters", 4))
        self.reset()

    def set_elbow_min(self, val: float) -> float:
        """Live-set the elbow (j3) lower limit and rebuild the IK ConfigurationLimit,
        so you can dial out the wrong-way bend in real time. Clamped to the hard limit
        and below soft_hi. Returns the applied value."""
        e = self.ELBOW
        v = float(np.clip(val, self.hard_lo[e], self.soft_hi[e] - 0.05))
        self.soft_lo[e] = v
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.joints[e])
        self.model.jnt_range[jid] = [v, self.soft_hi[e]]
        clim = mink.ConfigurationLimit(self.model)   # caches ranges -> must rebuild
        self.limits_pos[0] = clim
        self.limits_ori[0] = clim
        return v

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

    def limit_margins(self, q: np.ndarray | None = None) -> np.ndarray:
        """Per-joint signed distance (rad) to the nearest SOFT limit — for the HUD
        ('highlight any joint within X% of a limit', spec Section 6) and tests.
        Positive = inside the soft band; ~0 = at a limit; negative = past it."""
        q = self.q if q is None else np.asarray(q, dtype=float)
        return np.minimum(q - self.soft_lo, self.soft_hi - q)

    def within_limits(self, q: np.ndarray | None = None, tol: float = 1e-6) -> bool:
        """True iff every joint is within its soft limits (no buckle / no elbow
        hyperextension). Used by the synthetic harness to assert invariant #5."""
        return bool(np.all(self.limit_margins(q) >= -tol))

    def _joint_axis_base(self, name: str) -> np.ndarray:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return self.config.data.xaxis[jid].copy()   # joint axis in the (base) world frame

    def ee_semantic_frame_local(self) -> np.ndarray:
        """The EE site's own axes, expressed in EE-LOCAL coords, labelled by what the
        operator's wrist does: forward = j6 ROLL axis, right = j5 PITCH axis, up =
        their cross. Calibration aligns the operator's hand frame onto THIS, so a
        wrist twist drives j6 (not a forearm arc on j4). Columns = [right, up, forward]."""
        self.config.update(self.q0)
        mujoco.mj_forward(self.model, self.config.data)
        f = self._joint_axis_base(self.joints[5])              # j6 = roll/approach axis
        r = self._joint_axis_base(self.joints[4])              # j5 = pitch axis
        f = f / (np.linalg.norm(f) + 1e-12)
        r = r - (r @ f) * f
        r = r / (np.linalg.norm(r) + 1e-12)                    # orthonormalize against forward
        # u = r × f (NOT f × r): the operator/world frames (calibrate.W_AXES,
        # head_op_axes) are built LEFT-handed, so this EE frame must use the SAME
        # handedness or the calibrated correspondence P = E_loc @ Lᵀ comes out a
        # REFLECTION (det −1) and the wrist twist maps mirrored — the flipped-wrist bug.
        u = np.cross(r, f)
        ee_R = self.fk_ee().rotation().as_matrix()             # EE → base
        E_base = np.column_stack([r, u, f])
        return ee_R.T @ E_base                                 # → EE-local

    def solve(self, target: mink.SE3, iters: int | None = None) -> np.ndarray:
        iters = self.iters if iters is None else iters
        self.pos_task.set_target(target)
        self.ori_task.set_target(target)
        for _ in range(iters):   # stage 1: position → arm joints j1-j3
            v = mink.solve_ik(self.config, [self.pos_task, self.posture], self.dt,
                              self.solver, damping=self.damping, limits=self.limits_pos)
            self.config.integrate_inplace(v, self.dt)
        for _ in range(iters):   # stage 2: orientation → wrist joints j4-j6
            # posture (weak) pulls the free wrist joints toward home, so a pure ROLL
            # is taken by j6 (the hand-axis roll joint) instead of swinging j4 — j4
            # arcs the hand, which is the "arm jerks upward on wrist twist" symptom.
            v = mink.solve_ik(self.config, [self.ori_task, self.posture], self.dt,
                              self.solver, damping=self.damping, limits=self.limits_ori)
            self.config.integrate_inplace(v, self.dt)
        return self.config.q.copy()
