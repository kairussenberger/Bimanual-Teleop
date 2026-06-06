# Official Resting Position — DO NOT CHANGE

This is the **frozen, canonical rest/home pose** of the bimanual YAM + ORCA robot.
It was approved by the operator on 2026-06-05 and must not be altered. If any tool,
calibration, or refactor changes these values, it is a regression — revert it.

The pose: a humanoid torso with both arms hanging straight down at the sides
(gravity-ragdoll), wrists directly under the shoulders, and **both palms rolled to
face inward toward the central metal shaft**.

## Frozen values (source of truth = `config/rig.yaml`)

Arm home joint angles `neutral_q` (rad), order `[j1, j2, j3, j4, j5, j6]`:

| arm   | neutral_q                                          |
|-------|----------------------------------------------------|
| left  | `[3.137, -0.004,  0.305, -0.162, -0.003, -1.571]`  |
| right | `[3.14,  -0.001,  0.305, -0.152,  0.001,  1.571]`  |

- `j1, j2` ≈ straight down → arm hangs vertical, wrist (j4) directly below the shoulder.
- `j3` ≈ 0.305 slight elbow bend (keeps the IK out of the full-stretch singularity).
- `j6` = ∓1.571 (±90°) → **palms face inward to the shaft**. left j6 = −1.571
  (CCW from top), right j6 = +1.571 (CW from top). Flipping these signs makes the
  palms face OUTWARD — that is wrong.

Supporting geometry (also frozen, all in `config/rig.yaml` / `sim/model.py`):

- Arm bases bolted to Kai's real elongated AgileX frame (Orca-Yam-teleop c2814b4,
  +0.5 m vs the old stand), so the shoulders ride at z ≈ 1.19:
  `arms.left.base_pos  = [-0.0248, -0.1700, 1.1908]`
  `arms.right.base_pos = [ 0.0101,  0.0801, 1.1875]`
- Stand = the real 6-part frame mesh (1230 mm tall) in
  `sim/model.py::_base_scene_xml` (`<!-- Real ELONGATED AgileX frame -->`); the
  earlier box-riser hack is gone. neutral_q (joint angles) is unchanged by the
  lift — raising the base just translates the whole arm up, so the hang is identical.
- Workspace box lowered so the arms-down target is not clipped on teleop engage
  (which had curled both arms to center): `safety.workspace.min[1] = -0.85`
  (was -0.2). Do not raise it back above ≈ -0.6.

## How it was derived (so it can be reproduced, not re-guessed)

Not hand-tuned: gravity was enabled in the sim with gravity-comp and actuator
holding-torque zeroed, the arms were let fall limp and settle, and the settled
qpos was read off — i.e. literally "how the arms fall." Then j6 was rolled ±90°
for the palms-inward requirement. Verified by render; operator-confirmed.

## Invariants any change must preserve

1. Wrist hangs ~directly under the shoulder (vertical arm).
2. Hands do NOT meet/curl at the center, and do NOT splay outward — straight down.
3. Palms face inward toward the central shaft.
4. Engaging teleop does not move the arms from rest (workspace box must contain the
   home wrist: `clip_shift == 0`).
5. `python tests/test_pipeline.py` stays 16/16.
