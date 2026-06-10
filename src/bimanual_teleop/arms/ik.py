"""Per-arm differential IK for the 6-DoF YAM, using **pink** (Pinocchio QP diff-IK)
on a STANDALONE single-arm model (so one arm's IK can never perturb the other).
Poses are in the arm's base frame; the kinematics come from arms/yam_pin (transcribed
from the measured MJCF source geometry). This replaces the former
mink/MuJoCo backend — same two-stage algorithm, same joint/velocity limits, same
posture task; only the solver engine changed (invariant #2: still a real QP diff-IK,
never a hand-rolled pseudoinverse).

THREE-STAGE solve (the key to clean teleop on this arm):
  1. POSITION — move j1-j3 (the arm) to the target wrist position. j4-j6 frozen.
  2. TWIST — the roll component of the remaining orientation error about the
     CURRENT j6/tool axis is computed analytically (swing–twist decomposition)
     and assigned DIRECTLY to j6, rate-limited and clamped to its limits. Pure
     wrist roll therefore lands on j6 BY CONSTRUCTION, and roll beyond j6's
     range SATURATES instead of being smeared onto j4/j5 — chasing unreachable
     roll with the wrist QP is what used to fold the wrist through its
     singularity.
  3. SWING — a QP step on j4-j5 only (j1-j3 and j6 frozen) re-aims the tool
     axis for the residual orientation, with the unrealizable twist remainder
     REMOVED from the target so the swing joints never try to fake a roll.
The freezes are implemented by mutating the model's per-joint velocity limit
between stages (pink's VelocityLimit re-reads it each solve). The analytic twist
is an exact 1-DoF assignment to the motor whose axis IS the twist axis — the
swing/position stages remain real QP diff-IK (invariant #2).

JOINT LIMITS & COLLISIONS (spec invariant #5 + Section 8):
  - Hard limits live on the Pinocchio model (lower/upperPositionLimit) AND are enforced
    in the IK via pink.ConfigurationLimit. On top, SOFT limits cap each joint to home ±
    margin so the arm physically cannot buckle or HYPEREXTEND THE ELBOW (j3). See
    limit_margins() for the live per-joint margin.
  - Self-collision avoidance: this IK runs on a STANDALONE single-arm model, so it
    cannot see the other arm/torso — cross-arm collision is mitigated upstream by the
    anti-cross world-Y guard + workspace box in arm_control. (pink supports collision
    barriers, but they need a Pinocchio GeometryModel we don't build for the standalone
    arm; passing collision_pairs is therefore rejected until a collidable model exists.)
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin
import pink
from pink.tasks import FrameTask, PostureTask
from pink.limits import ConfigurationLimit, VelocityLimit

from ..vr.frames import SE3, R_to_quat, quat_to_R
from .yam_pin import build_arm_model, joint_local_axis


def swing_twist_angle(R_err: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle of the TWIST component of R_err about unit `axis`, from the
    swing–twist decomposition R_err = R_swing · R_twist(axis, angle). Wrapped to
    [-π, π]; 0 when R_err has no component about the axis."""
    q = R_to_quat(R_err)
    if q[0] < 0.0:
        q = -q                                  # shortest-arc representation
    proj = float(q[1:] @ axis)
    n = float(np.hypot(q[0], proj))
    if n < 1e-12:
        return 0.0
    ang = 2.0 * np.arctan2(proj / n, q[0] / n)
    return float((ang + np.pi) % (2.0 * np.pi) - np.pi)

class ArmIK:
    ELBOW = 2   # index of j3, the elbow joint (soft limit caps its hyperextension)

    def __init__(self, rig: dict, side: str, collision_pairs=None):
        if collision_pairs:
            raise NotImplementedError(
                "collision avoidance needs a Pinocchio GeometryModel; the standalone "
                "arm model has no geometry. Cross-arm collision is handled upstream "
                "(anti-cross Y guard + workspace box in arm_control).")
        self.side = side
        self.joints = [f"{side}_arm_j{i}" for i in range(1, 7)]   # 6-DoF
        ik = rig["ik"]
        self.q0 = np.asarray(rig["arms"][side]["neutral_q"], dtype=float)
        self.max_vel = float(ik["max_vel"])
        self.froze = 1e-4   # near-zero velocity to freeze a joint (above pink's 1e-10 cutoff)

        self.model = build_arm_model(side, max_vel=self.max_vel)
        # joint index → velocity-vector slot (j1..j6, all revolute → idx_v 0..5)
        self._idx_v = [self.model.joints[self.model.getJointId(j)].idx_v for j in self.joints]
        self._idx_q = [self.model.joints[self.model.getJointId(j)].idx_q for j in self.joints]
        self.arm_v = self._idx_v[:3]
        self.wrist_v = self._idx_v[3:]

        # SOFT joint limits = home ± margin (clamped to hard limits) so the IK physically
        # cannot fold the arm into a buckle. Written onto the model so ConfigurationLimit
        # reads them.
        self.hard_lo = np.asarray(rig["arms"]["joint_limits"]["lower"], dtype=float)
        self.hard_hi = np.asarray(rig["arms"]["joint_limits"]["upper"], dtype=float)
        margin = np.asarray(ik.get("soft_margin", [1.4, 1.0, 1.4, 1.4, 1.5, 1.7]), dtype=float)
        self.soft_lo = np.maximum(self.q0 - margin, self.hard_lo)
        self.soft_hi = np.minimum(self.q0 + margin, self.hard_hi)
        # Human-plausible ELBOW (invariant: no overextension). j3 is straightest at home
        # and bends BOTH ways, but only +j3 is anatomical flexion; −j3 swings the forearm
        # outward into a dislocated-looking bend. Floor j3 so it can't go there.
        emin = ik.get("elbow_min", None)
        if emin is not None:
            self.soft_lo[self.ELBOW] = max(self.soft_lo[self.ELBOW], float(emin))
        emax = ik.get("elbow_max", None)
        if emax is not None:
            self.soft_hi[self.ELBOW] = min(self.soft_hi[self.ELBOW], float(emax))
        self._apply_pos_limits()

        # Tasks. POSITION targets the WRIST frame (before the wrist joints → invariant to
        # j4/j5/j6); ORIENTATION targets the hand (ee frame).
        self.pos_task = FrameTask(f"{side}_wrist", position_cost=ik["pos_cost"],
                                  orientation_cost=0.0, lm_damping=ik["lm_damping"])
        self.ori_task = FrameTask(f"{side}_ee", position_cost=0.0,
                                  orientation_cost=ik["ori_cost"], lm_damping=ik["lm_damping"])
        self.posture = PostureTask(cost=ik["posture_cost"])

        self.data = self.model.createData()
        self.config = pink.Configuration(self.model, self.data, self.q0.copy())
        # ONE ConfigurationLimit + ONE VelocityLimit, shared across both stages; the
        # two-stage freeze mutates model.velocityLimit in place between stages.
        self._build_limits()
        # back-compat aliases (tests inspect these); both stages share the same limits.
        self.limits_pos = self.limits
        self.limits_ori = self.limits

        self.solver = ik.get("solver", "daqp")
        self.damping = float(ik.get("damping", 1e-3))
        self.dt = 1.0 / rig["control"]["arm_hz"]
        self.iters = int(ik.get("iters", 4))
        self.reset()

    # ---- model limit plumbing --------------------------------------------- #
    def _apply_pos_limits(self) -> None:
        for i, iq in enumerate(self._idx_q):
            self.model.lowerPositionLimit[iq] = self.soft_lo[i]
            self.model.upperPositionLimit[iq] = self.soft_hi[i]

    def _build_limits(self) -> None:
        self.limits = [ConfigurationLimit(self.model), VelocityLimit(self.model)]

    def set_elbow_min(self, val: float) -> float:
        """Live-set the elbow (j3) lower limit and rebuild the IK ConfigurationLimit, so
        you can dial out the wrong-way bend in real time. Clamped to the hard limit and
        below soft_hi. Returns the applied value."""
        e = self.ELBOW
        v = float(np.clip(val, self.hard_lo[e], self.soft_hi[e] - 0.05))
        self.soft_lo[e] = v
        self.model.lowerPositionLimit[self._idx_q[e]] = v
        self._build_limits()   # ConfigurationLimit caches the projection -> must rebuild
        self.limits_pos = self.limits
        self.limits_ori = self.limits
        return v

    def reset(self) -> None:
        self.config.update(self.q0.copy())
        self.posture.set_target(self.q0.copy())

    def seed(self, q: np.ndarray) -> None:
        self.config.update(np.asarray(q, dtype=float))

    @property
    def q(self) -> np.ndarray:
        return self.config.q.copy()

    def fk_ee(self) -> SE3:
        return SE3.from_pin(self.config.get_transform_frame_to_world(f"{self.side}_ee"))

    def fk_wrist(self) -> SE3:
        return SE3.from_pin(self.config.get_transform_frame_to_world(f"{self.side}_wrist"))

    def link_points(self) -> np.ndarray:
        """Polyline points for renderers, in this standalone arm base frame.

        Uses the live Pinocchio model state instead of duplicating kinematic
        constants in render clients. Points are: base origin, j1..j6 origins, ee.
        """
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        pts = [np.zeros(3)]
        for name in self.joints:
            pts.append(np.asarray(self.data.oMi[self.model.getJointId(name)].translation).copy())
        pts.append(self.fk_ee().translation())
        return np.stack(pts)

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
        """Joint rotation axis in the base frame at the CURRENT config (the Pinocchio
        equivalent of mink's config.data.xaxis[jid]). `name` like '{side}_arm_j6'."""
        jid = self.model.getJointId(name)
        a = joint_local_axis(name.split("_arm_")[1])
        return self.config.data.oMi[jid].rotation @ a

    def _set_stage_velocity(self, free_arm: bool) -> None:
        """Freeze one half of the chain by writing the model velocity limit, then
        reassign it as a whole (eigenpy property write-back is reliable that way)."""
        vl = np.full(self.model.nv, self.max_vel)
        if free_arm:
            vl[self.wrist_v] = self.froze   # stage 1: arm free, wrist frozen
        else:
            vl[self.arm_v] = self.froze     # legacy full-wrist stage (kept for API)
        self.model.velocityLimit = vl

    def _set_swing_velocity(self) -> None:
        """Stage 3: only the swing joints j4/j5 free; arm AND j6 frozen (the twist
        was already assigned analytically)."""
        vl = np.full(self.model.nv, self.froze)
        vl[self.wrist_v[0]] = self.max_vel
        vl[self.wrist_v[1]] = self.max_vel
        self.model.velocityLimit = vl

    def _apply_twist(self, target_R: np.ndarray, budget: float) -> np.ndarray:
        """Stage 2: assign the roll component of the orientation error directly to
        j6 (rate-limited, limit-clamped) and return the stage-3 SWING target with
        the unrealizable twist remainder removed."""
        R_c = self.fk_ee().rotation().as_matrix()
        a = self._joint_axis_base(self.joints[5])
        a = a / (np.linalg.norm(a) + 1e-12)
        R_err = target_R @ R_c.T
        phi = swing_twist_angle(R_err, a)
        iq = self._idx_q[5]
        q = self.config.q.copy()
        q5_new = float(np.clip(q[iq] + np.clip(phi, -budget, budget),
                               self.soft_lo[5], self.soft_hi[5]))
        applied = q5_new - float(q[iq])
        if abs(applied) > 1e-12:
            q[iq] = q5_new
            self.config.update(q)
        # R_err = R_swing · Rot(a, φ)  ⇒  swing-only residual target excludes the
        # twist we could not (or chose not yet to) apply:
        rot = lambda ang: quat_to_R([np.cos(ang / 2.0), *(np.sin(ang / 2.0) * a)])
        return R_err @ rot(applied - phi) @ R_c

    def solve(self, target: SE3, iters: int | None = None) -> np.ndarray:
        iters = self.iters if iters is None else iters
        tgt = target.to_pin()
        self.pos_task.set_target(tgt)
        self._set_stage_velocity(free_arm=True)    # stage 1: position → arm joints j1-j3
        for _ in range(iters):
            v = pink.solve_ik(self.config, [self.pos_task, self.posture], self.dt,
                              self.solver, damping=self.damping, limits=self.limits,
                              safety_break=False)
            self.config.integrate_inplace(v, self.dt)
        # stage 2: TWIST → j6 directly (rate budget matches what the QP stages get)
        target_R = target.rotation().as_matrix()
        swing_R = self._apply_twist(target_R, budget=self.max_vel * self.dt * iters)
        self.ori_task.set_target(SE3.from_rotation_and_translation(
            swing_R, target.translation()).to_pin())
        self._set_swing_velocity()                 # stage 3: SWING → j4/j5 only
        for _ in range(iters):
            # posture (weak) pulls the free swing joints toward home.
            v = pink.solve_ik(self.config, [self.ori_task, self.posture], self.dt,
                              self.solver, damping=self.damping, limits=self.limits,
                              safety_break=False)
            self.config.integrate_inplace(v, self.dt)
        return self.config.q.copy()
