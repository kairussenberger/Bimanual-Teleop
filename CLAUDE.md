# Teleop Runtime Notes

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

## Orientation Mapping Contract

EE orientation is CALIBRATION-FREE and uses the same body→world axis map as
translation (`ClutchMapper.target`): the wrist rotation since clutch-engage,
measured in body axes, is applied to the anchored EE orientation about the
corresponding robot-world axes — rotate your hand θ about body-forward and the EE
rotates θ about world −X, from any starting pose. `vr.calib_seconds` defaults to 0;
the legacy stance hold only steers arms when `vr.body_relative` is false.

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
2. Solve hand orientation with j4-j6 while j1-j3 are velocity-limited near zero.
3. Enforce soft limits around the configured rest pose, including the elbow floor.
4. Keep pure wrist roll on j6.

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
