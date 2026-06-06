# VR Teleoperation Build Spec — Quest → Bimanual MuJoCo Rig

A build roadmap and architecture spec for a Quest-driven bimanual teleop system,
written to be handed to Claude Code. Work through it in the **bring-up order**
in Section 8 — do not build everything at once.

---

## 0. System context (read first)

- **Robot:** two 6-DOF arms (I2RT YAM) mounted on a shared torso/T-frame, with an
  ORCA hand on the right (and a hand/gripper on the left). Eventually deploys on
  hardware (YAM over SocketCAN on Ubuntu); for now everything runs in **MuJoCo**.
- **Input:** Meta Quest. The ONLY signal available for arm control is a **6-DOF
  wrist pose per hand** — position + orientation. No finger joints are used to
  drive the arms. Treat each hand as: `(p_hand, q_hand, tracked_flag)`.
- **Goal:** the operator's wrist motion is mirrored by the corresponding arm so it
  feels like *direct control*. Position mapping mostly works today; **wrist
  orientation (roll about the forearm axis) is the thing that fails.**
- **Known constraints to respect:**
  - A human cannot hyperextend the elbow → the robot must not either.
  - The ORCA wrist DOF is **flexion/extension only** (±60°, radioulnar axis). It
    has **no pronation/supination**. Therefore ALL forearm roll
    (pronation/supination) must be produced by the **arm's terminal roll joint
    (J6)**. This is central to the orientation bug — see Section 3.

## 1. Non-negotiable architecture invariants

These are the rules that, if violated, produce the "uncontrollable / never deploy
this" behavior. Put this list verbatim in `CLAUDE.md` (Section 9) so every coding
session respects it.

1. **Relative, clutched mapping — never absolute.** The robot target is an offset
   from a snapshot taken at clutch-engage, not a direct copy of the hand pose.
2. **Differential IK via `mink` — never a hand-rolled Jacobian pseudoinverse.**
   `mink` gives velocity-domain QP IK with joint position + velocity limits,
   posture tasks, and collision avoidance, warm-started from the current config
   (so solutions are continuous — no elbow/wrist flips).
3. **All tracking input is filtered** (One-Euro) before it reaches IK.
4. **Every frame transform is explicit and named.** No implicit "it happens to line
   up." There is exactly one `R_align` (headset→robot basis change), applied
   consistently to position deltas and orientation deltas.
5. **Joint limits are enforced** in both the MuJoCo model and the IK
   (`ConfigurationLimit`), including a human-plausible elbow limit.
6. **The control loop runs at a fixed rate, latest-pose-wins**, with networking
   decoupled from control. Stale frames are dropped, not queued.
7. **Observability is a first-class feature, not an afterthought** (Section 6). At
   any instant the operator can see: tracking status, clutch state, calibration
   state, commanded vs achieved EE frames, and joint-limit/velocity margins.
8. **Tracking loss freezes the robot** and shows a clear warning. Never extrapolate
   blindly through a dropout.

## 2. Coordinate frames & conventions

Define and name every frame. Centralize them in `config/frames.yaml` and a single
`frames.py` module. Frames:

| Frame | Meaning |
|---|---|
| `world` | MuJoCo world frame (Z up, right-handed). |
| `robot_base` | Torso/T-frame root; arms branch from here. |
| `ee_site_L`, `ee_site_R` | MuJoCo sites at each arm's end effector (the IK targets). Add these sites to the model XML if absent. |
| `headset_world` | Quest/OpenXR tracking-space frame (right-handed, **+Y up, −Z forward, +X right**). |
| `hand_raw_L/R` | Wrist pose as reported by the Quest, in `headset_world`. |
| `hand_filt_L/R` | One-Euro-filtered wrist pose. |

**The basis change `R_align`.** OpenXR (`+Y` up) and MuJoCo (`+Z` up) do **not**
share an axis convention. You must apply a fixed rotation `R_align` that maps
vectors/rotations from `headset_world` into `robot_base`. This is almost certainly
why translation "kinda works" but rotation "fails completely": position is
forgiving (a wrong axis just sends the arm sideways, still legible), but a wrong
basis scrambles a pure roll into a mix of pitch+yaw, so wrist rotation looks
incoherent. **Determine `R_align` explicitly, store it in config, and verify it
with the synthetic roll test (Section 8, step 3).**

## 3. The mapping math (the part that's failing)

Let engage-time snapshots be `p_h0, q_h0` (hand in `headset_world`) and
`x_e0, q_e0` (EE in `robot_base`). Let `q_align` be the quaternion of `R_align`.

**Position (relative, optionally scaled):**

```
dp_head   = p_hand(t) - p_h0           # delta in headset frame
dp_robot  = R_align @ dp_head * scale  # rotate into robot frame
x_target  = x_e0 + dp_robot
```

`scale = 1.0` for 1:1 mirroring; expose it as a tunable (use < 1 for fine work).

**Orientation (relative — do this carefully):**

```
dq_head  = q_hand(t) * inv(q_h0)              # hand delta rotation, in headset axes
dq_robot = q_align * dq_head * inv(q_align)   # CHANGE OF BASIS into robot axes
q_target = dq_robot * q_e0                    # apply as world-frame delta to engaged EE
```

The conjugation `q_align * dq_head * inv(q_align)` is the step people forget — a
rotation expressed in one frame must be **conjugated**, not just multiplied, to be
expressed in another frame. Get the quaternion multiplication order right and be
explicit about world-frame (left-multiply) vs body-frame (right-multiply) deltas.
Add a unit test: feed a known +90° roll about the hand's forward axis and assert
`q_target` is a +90° roll about the robot's tool axis — nothing else.

**Pronation/supination → J6.** With a correct `R_align`, the forearm-roll component
ends up as a roll about the EE tool axis, which `mink` will realize on J6 (since
ORCA can't). Two checks: (a) the EE task must include **full orientation**, not
just position; (b) confirm the YAM J6 joint range covers human roll (~150–180°
total) — if it's narrower, add an orientation re-center to the clutch so the
operator can "regrip" roll like repositioning a mouse.

## 4. Calibration

Your current "hands in front → robot matches, then move around" is the right idea
but under-specified. Make it a small state machine with these states, each
**observable** (Section 6): `IDLE → COUNTDOWN → CAPTURING → SOLVING → ACTIVE`, plus
`FAILED`.

- **Engage snapshot:** capture `p_h0,q_h0` and `x_e0,q_e0` for each arm at the
  moment the operator strikes the calibration pose and confirms. These define the
  relative-mapping origin.
- **Estimating `R_align`:** start with the static OpenXR→MuJoCo basis change plus a
  per-session yaw correction (operator facing direction). Optionally upgrade to a
  multi-pose solve: have the operator hold 3–4 distinct wrist orientations, log
  hand vs intended robot orientations, and least-squares fit the rotation. Report
  the residual; if it's large, go to `FAILED` and tell the operator why.
- **Recalibratable on the fly** with a button; never require a restart.
- **Sanity checks:** after calibration, verify a small commanded motion produces a
  small EE motion in the expected direction; warn loudly if not.

## 5. The control loop

Fixed-rate (start 60–100 Hz), with a separate thread/process reading the Quest and
publishing the latest pose into a single-slot buffer (latest-wins).

Per tick, in order:
1. Read latest `(p_hand, q_hand, tracked)` for each hand; if stale (age > threshold)
   mark untracked.
2. If any required hand is untracked → **hold** current joint targets, raise a
   visible warning, skip to logging.
3. One-Euro filter the pose(s).
4. If clutch disengaged → hold; if just engaged → take the engage snapshot.
5. Compute `x_target, q_target` per Section 3.
6. Set the per-arm `mink` EndEffectorTask targets.
7. Solve the QP (limits + posture + collision), integrate to get `q_des`.
8. Apply `q_des` as position targets to the MuJoCo actuators; step the sim.
9. Log telemetry + update visualization overlays (Section 6).

## 6. Observability — logging & visualization (build this EARLY, not last)

This is the cure for "I'm flying blind." Two layers:

**A. In-sim overlays (MuJoCo viewer):**
- Per arm, draw the **commanded target frame** and the **achieved EE frame** as
  color-coded triads. Seeing them drift apart instantly tells you lag vs offset vs
  flip.
- Draw the raw hand triad and the filtered/aligned hand triad.
- On-screen HUD text panel showing: per-hand tracking status (TRACKED / STALE /
  LOST), clutch state, calibration state + residual, loop rate (Hz), IK solve time,
  IK success flag, and pose error (position cm + orientation deg) per arm.
- Joint state bars: highlight any joint within X% of a limit; flag velocity
  saturation; show a near-singularity / low-manipulability warning.

**B. Structured logging + live dashboard:**
- Use **Rerun** (`rerun.io`) as the telemetry backbone — it is built for exactly
  this: log timestamped 3D transforms (all the frames above), scalar time-series
  (pose error, loop rate, solve time, joint margins), and text/status, then scrub
  and replay. This gives you the "what is MuJoCo actually doing" view you're
  missing.
- Also keep a structured Python logger (levelled) with clear human-readable
  warnings: `WARN left-hand tracking LOST`, `WARN calibration residual 14° (high)`,
  `WARN J3 at limit`, `WARN IK did not converge`.
- Dump per-tick telemetry to a ring buffer and optionally to CSV/Parquet for
  offline plotting.
- Signals to log every tick: t, loop_dt, pose age/latency, raw + filtered hand
  pose, clutch state, calib state, x_target/q_target, achieved EE pose, pos error,
  rot error, q, qvel, per-joint limit margin, IK solve time, IK status, velocity
  saturation flags.

## 7. Test harness & debugging modes

- **Synthetic-input mode:** drive EE targets with scripted trajectories (line,
  circle, **pure roll**, pure pitch, pure yaw) with no headset attached. This is
  how you isolate the J6 bug: if synthetic pure-roll tracks cleanly, the problem is
  frames/tracking; if it doesn't, it's IK/limits.
- **Replay mode:** record a Quest session to disk and replay it deterministically
  through the full pipeline, so you can debug without wearing the headset.
- **Unit tests** for: every frame transform (known-rotation assertions), the
  One-Euro filter, clutch snapshot/relative logic, and the calibration solver.

## 8. Bring-up order (milestones — do them in sequence)

1. MuJoCo model loads; add `ee_site_L/R`; render frame triads + HUD skeleton.
2. `mink` dual-arm IK driven by a **synthetic** EE target; verify smooth tracking,
   joint + velocity limits hold, posture task keeps a human-like elbow, no flips.
3. **Synthetic pure rotations** (roll/pitch/yaw) — get J6/orientation correct here,
   before any headset is involved.
4. Add Quest input; log + visualize the raw hand triads; confirm tracking flags and
   stale detection.
5. Add `R_align` + calibration; verify orientation mapping with the overlays (the
   synthetic roll test from step 3 should now reproduce from a real wrist roll).
6. Add clutch + relative mapping (position then orientation).
7. Add One-Euro filtering; tune scale, filter, and velocity limits for feel.
8. Enable bimanual + self-collision avoidance between arms/torso.
9. Hardware bring-up (YAM SocketCAN, ORCA Dynamixel) — only after sim feels
   deployable.

## 9. Repo layout & VS Code / Claude Code workflow

Suggested structure (uv-managed):

```
teleop/
  config/        frames.yaml, limits.yaml, gains.yaml, calibration.yaml
  teleop/
    transport.py     # Quest interface, latest-wins buffer, tracking flags
    frames.py        # all transforms, R_align, change-of-basis helpers
    filters.py       # One-Euro
    calibration.py   # state machine + R_align solve
    ik.py            # mink wrapper: tasks, limits, posture, collision
    control_loop.py  # fixed-rate loop
    viz.py           # MuJoCo overlays + Rerun logging
    logging_utils.py # structured logger, telemetry ring buffer
    replay.py        # record/replay
  scripts/       run_sim.py, run_synthetic.py, calibrate.py, replay.py
  tests/         test_frames.py, test_filters.py, test_clutch.py, test_calib.py
  CLAUDE.md
  pyproject.toml
```

**`CLAUDE.md`** (repo root) should contain: the Section 0 context, the Section 1
invariants verbatim, the bring-up order, and a rule that frame transforms get a
unit test before use. This keeps every Claude Code session on-architecture instead
of regenerating the naive loop.

**Working style with Claude Code:** tackle one milestone at a time; ask it to write
the transform tests first; never let it replace `mink` with a hand-rolled solver or
relative mapping with absolute. If a change makes the feel worse, revert and bisect
with replay mode rather than re-prompting from scratch.

## 10. Reference implementations to lean on (don't reinvent)

- **mink** — `github.com/kevinzakka/mink` — differential IK on MuJoCo; QP with
  position/velocity limits, posture tasks, collision avoidance; dual-arm examples.
  Used in DexMimicGen for a single-torso bimanual humanoid — your exact topology.
- **OpenTeleVision** — reference for the relative-mapping + filtering + Pinocchio IK
  teleop loop and stereo feedback; good to crib the structure from.
- **One-Euro filter** — small, well-documented; copy a reference implementation.
- **Rerun** (`rerun.io`) — the observability backbone described in Section 6.
- **dex-retargeting** — for the ORCA finger retargeting path (separate from the arm
  loop covered here).