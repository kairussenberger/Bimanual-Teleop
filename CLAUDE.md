# Teleop Runtime Notes

System overview, failsafe inventory, tooling index, and the sim→real checklist
live in `docs/ARCHITECTURE.md` — keep that page current when changing any of
them. The hardware boundary (`HardwareSink`) must always command through
`safety/shaper.py` (limit-clamp + speed cap + PD smoothing); never bypass it.

This repo no longer uses MuJoCo as the runtime simulator. The current target is a
headless Python teleop process that:

- ingests Quest/ORBIT, Vuer/WebXR, replay, or synthetic poses;
- converts raw headset/hand poses into body-relative torso-to-wrist vectors;
- solves each YAM arm with a standalone Pinocchio/pink IK model;
- retargets ORCA hand landmarks to hardware joint degrees;
- publishes `render.state` for Unity over ZMQ/msgpack and plain TCP JSON;
- can swap the render sink for the real hardware sink on the Linux robot host.

The remaining MJCF files under `src/bimanual_teleop/sim/models/yam_real/` are
source geometry for the programmatic Pinocchio model in
`src/bimanual_teleop/arms/yam_pin.py`. They are not loaded by the runtime.

## Coordinate Contract

Robot world is right-handed:

- `+Z` is up.
- `+Y` is operator/robot right.
- `-X` is forward.

Quest/ORBIT Unity poses are converted before they reach the teleop engine. Arm
control then uses body-relative wrist samples:

```text
raw head + raw wrist -> operator body axes -> torso proxy -> wrist_body
```

`wrist_body` is `[right, up, forward]` from the torso proxy to the wrist. Whole
head/body translation should cancel. A hand lift should appear as a positive
`wrist_body[1]` delta and drive the corresponding arm target upward.

The Unity render stream exposes the same vector at
`render.state.op.hands.*.wrist_body` so the operator overlay and the robot arm
state can be compared directly.

## Motion Mapping Contract

POSITION is ABSOLUTE and body-anchored (`mapping.position_mode: absolute`): the
operator's torso→wrist vector maps 1:1 (×`pos_scale`) onto the robot's
chest→wrist vector, where the chest anchor defaults to the midpoint of the arm
bases dropped by `body_anchor_drop`. Hands held in front of the operator put the
robot's wrists in front of the robot — verified at 0.1 cm median correspondence
on a real session. On (re)engage the EE GLIDES onto correspondence over
`mapping.engage_blend_s` (continuous at the engage instant; displacements map 1:1
from the first tick). The `ik.soft_margin` values are sized so this front-of-body
workspace is reachable — re-tightening them below the measured excursions in
config/rig.yaml's comment will pin the arm short of its targets again.

EE ORIENTATION is CALIBRATION-FREE and split swing/twist
(`mapping.twist_mode: intrinsic`, `ClutchMapper.target`): the wrist rotation
since clutch-engage is decomposed about the operator's forearm axis
(`mapping.hand_twist_axis`, measured from a real session). The TWIST becomes an
EE roll about the EE's OWN tool/j6 axis — a wrist turn is always a pure j6 roll,
never a j4/j5 swing through the wrist singularity (verified at 1.9° median
contract error on a real recording). The residual SWING (real pitch/yaw of the
hand) maps through the same body→world axes as translation. `twist_mode: world`
(fully extrinsic) is diagnostics-only. `vr.calib_seconds` defaults to 0; the
legacy stance hold only steers arms when `vr.body_relative` is false.

Known physics: orientation is relative-latched, so a large attitude offset
between hand and EE accumulated at engage (e.g. engaging at rest then raising
the hand without re-orienting) can command wrist attitudes near the YAM's
reach/limit envelope; re-engaging the clutch re-anchors and clears the offset.
If live feel demands it, the designed next step is absolute orientation with a
fixed hand↔EE convention (derivable from the measured hand axes — still no
stance calibration).

Do not reintroduce a stance-calibrated hand-local↔EE-local correspondence: an
imperfect calibration pose scrambles every commanded rotation axis (measured 145°
median axis error on a real Quest session; `tests/test_frames.py::
test_clutch_orientation_body_relative_real_rig_axes` pins the contract). Note that
`head_op_axes`/`W_AXES` are left-handed `[right, up, forward]` bases (det −1); the
ClutchMapper conjugation cancels the two reflections, so keep orientation math
going through BOTH or neither.

To grade a recorded session against this contract (axis error, angle ratio,
translation direction, IK tracking gap — no headset needed):

```sh
uv run python scripts/analyze_session.py recordings/session.npz
```

To watch it in 3D without Unity (`uv sync --extra telemetry` once):

```sh
uv run python -m bimanual_teleop.launch.run_teleop --vr replay session.npz --viz
```

## IK Contract

Arm IK lives in `src/bimanual_teleop/arms/ik.py` and uses Pinocchio/pink:

1. Solve wrist position with j1-j3 while j4-j6 are velocity-limited near zero.
2. Assign the TWIST (roll about the current j6/tool axis) of the orientation
   error DIRECTLY to j6 via the analytic swing–twist decomposition —
   rate-limited and clamped, so roll beyond j6's range saturates gracefully.
3. Solve the residual SWING with j4-j5 only (j6 frozen), with the unrealizable
   twist remainder removed from the target so the swing joints never contort
   through the wrist singularity to fake a roll.
4. Enforce soft limits around the configured rest pose, including the elbow floor.

Pure wrist roll lands on j6 BY CONSTRUCTION
(`tests/test_ik.py::test_roll_beyond_j6_range_saturates_without_contortion` pins
the saturation behavior). Mind the physical j6 range: ±120° motor with the frozen
rest at ∓90° leaves asymmetric roll headroom per side.

Do not reintroduce a hand-rolled Jacobian solver for arm control. If IK behavior
regresses, use `scripts/run_synthetic.py` first; it isolates line/circle/roll/pitch
and yaw targets without any headset or Unity dependency.

## Replay Contract

Use `run_teleop --record session.npz` to capture the exact head/wrist/finger stream
and engagement decisions for later debugging. `run_teleop --vr replay session.npz`
must drive the same `TeleopEngine`, `RenderSink`, and Unity JSON stream. Replay
sample selection uses recorded time, but `ReplaySource.latest()` refreshes
`VRFrame.stamp` to the current monotonic clock so the live staleness gate does not
drop valid recordings. `run_teleop --vr replay` uses the recorded engagement
decisions by default; overriding `--clutch` is for deliberate policy experiments.

## Unity Contract

`src/bimanual_teleop/render_sink.py` publishes robot state. Unity consumes the TCP
JSON stream by default because it needs only `TcpClient` and `JsonUtility`.

Arm geometry is authoritative in Python. The render payload includes
`arms.*.link_pos`, a flattened base, j1..j6, EE polyline from the live Pinocchio
state. Unity should convert and draw those points; it should not duplicate FK
constants.

See `docs/UNITY_BRIDGE.md` and `unity/TeleopRenderer/README.md`.

## Bring-Up Gates

Use this hardware-free acceptance gate after runtime changes:

```sh
uv run python scripts/verify_stack.py
```

It runs:

- `uv run pytest -q`
- the body-relative teleop probe (`scripts/check_body_relative.py`)
- the YAM source-geometry provenance check (`scripts/check_yam_geometry.py`)
- synthetic YAM trajectories
- the static Unity render contract and render-state fixture freshness check
- launch CLI parsing for `run_teleop` and `run_hw`
- a headless `run_teleop --vr fake` smoke
- a record/replay launch smoke
- a Unity JSON monitor smoke

On a machine with Unity installed, also run:

```sh
uv run python scripts/run_unity_validation.py --require
uv run python scripts/verify_stack.py --unity-editor
```

Unity Editor compilation and Quest rendering are external to this machine and must
be validated on a machine with Unity installed.
