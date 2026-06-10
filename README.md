# bimanual-teleop

VR teleoperation for a torso humanoid: two i2rt YAM arms with wrist-roll flanges
and two ORCA hands on an AgileX Ranger stand, driven by Meta Quest 3 / 3S hand
tracking.

The current runtime is MuJoCo-free. The Python process ingests headset/hand poses,
maps wrists to torso-relative arm targets, solves each YAM arm with a standalone
Pinocchio/pink model, retargets fingers to ORCA joints, and publishes a latest-value
render stream for a Unity Quest renderer. The same `TeleopEngine` can drive real
hardware by swapping the sink.

```text
Quest / synthetic / replay poses
  wrists + head pose -> calibration -> ArmController -> pink IK -> q[6]
  hand landmarks     -> HandController -> ORCA joint degrees
          |              |
          v              v
       Supervisor    sink.set_arm / set_hand
                         |-- RenderSink: ZMQ render.state for Unity
                         |-- HardwareSink: YAM CAN + ORCA serial on Linux
```

## Install

Requires Python 3.12 and `uv`. The ORCA hand control library is an editable sibling
dependency.

```sh
Developer/
├── orca_core/
└── bimanual-teleop/

cd bimanual-teleop
uv sync
uv sync --extra vr        # optional: Vuer/WebXR browser ingest
uv sync --extra telemetry # optional: Rerun dashboard for synthetic checks
```

## Quickstart

Run the full pipeline with a fake operator and publish Unity render frames:

```sh
uv run python -m bimanual_teleop.launch.run_teleop --vr fake
```

In another terminal, confirm the Unity-facing render stream:

```sh
uv run python scripts/render_monitor.py --seconds 5
uv run python scripts/render_monitor.py --transport json --require-hand-render --require-bimanual-state --require-command-target --require-frame --seconds 5
```

Run the synthetic IK verifier:

```sh
uv run python scripts/run_synthetic.py
```

Verify the runtime Pinocchio arm geometry still matches the source YAM MJCF body
trees:

```sh
uv run python scripts/check_yam_geometry.py
```

Run the explicit torso-relative mapping probe:

```sh
uv run python scripts/check_body_relative.py
```

Verify the default rig config still preserves body-relative Unity runtime defaults:

```sh
uv run python scripts/check_rig_contract.py
```

Run the hardware-free test suite:

```sh
uv run pytest -q
```

Run the full hardware-free acceptance gate:

```sh
uv run python scripts/verify_stack.py
```

On a machine with Unity Editor installed, run the Unity-side batch validation too:

```sh
uv run python scripts/run_unity_validation.py --require
uv run python scripts/verify_stack.py --unity-editor
```

## Real Quest Input

For the ORBIT native Quest app:

```sh
uv run python -m bimanual_teleop.launch.run_teleop --vr orbit --clutch gesture
```

The launcher sets `adb reverse` for the render channel when `--vr orbit` is used.
ORBIT pose input is converted from Unity coordinates into the internal WebXR-style
frame before calibration and relative wrist mapping. Use `scripts/run_hands.py` if
you only want to inspect hand tracking without running the robot pipeline. For
bring-up from the terminal, `scripts/check_quest.py` prints the torso-to-wrist
vector used by arm control, and `scripts/check_roll.py` analyzes wrist roll in the
same body-relative frame.

For browser WebXR/Vuer input:

```sh
uv run python -m bimanual_teleop.launch.run_teleop --vr vuer --clutch gesture
```

WebXR needs HTTPS certificates configured in `config/rig.yaml`.

## Record And Replay

Capture a problematic headset session, then replay the exact same head/wrist/finger
stream through the body-relative mapping and Unity renderer:

```sh
uv run python -m bimanual_teleop.launch.run_teleop --vr orbit --clutch gesture --record recordings/session.npz
uv run python -m bimanual_teleop.launch.run_teleop --vr replay recordings/session.npz
```

Replay uses fresh delivery timestamps for the safety supervisor, so recordings do
not immediately look stale when replayed later. `--vr replay` uses the recorded
engagement decisions by default; pass `--clutch always` or `--clutch gesture` only
when deliberately testing a different engage policy on the same motion.

## Debugging Without A Headset

You do not need the Quest (or Unity) to see and measure what the engine is doing.

Local 3D viewer (Rerun; works with fake/synthetic/replay/live sources):

```sh
uv sync --extra telemetry            # one-time: installs the Rerun SDK
uv run python -m bimanual_teleop.launch.run_teleop --vr replay recordings/session.npz --viz
uv run python -m bimanual_teleop.launch.run_teleop --vr fake --viz
```

The viewer shows both arm link chains, the achieved EE triad, the commanded EE
target triad, and your torso-to-wrist vector + wrist orientation mapped into robot
world by the SAME axes arm control uses — if the robot disagrees with that overlay,
the bug is in the mapping, not the IK. Error plots (`err/*`) and clutch/pinch
scalars come along for free, and with `--vr replay` the Rerun timeline scrubs the
whole session. `--viz-save out/session.rrd` logs the same scene to a file instead
of opening a window.

Watchable movie (no viewer app at all — renders a GIF/PNG of both tracked hands
next to the robot with real YAM mesh geometry, through the real engine):

```sh
uv run --with matplotlib python scripts/render_session.py recordings/session.npz --gif out/session.gif
```

Mapping scorer (replays a recording through the real engine and grades it):

```sh
uv run python scripts/analyze_session.py recordings/session.npz --side right
```

It reports, per the motion-mapping contract: the world-axis error between your
hand rotation and the commanded EE rotation, the translation direction error, and
the IK tracking gap — separating "mapping wrong" from "solver lagging" with
numbers. The scrambled-wrist bug showed up here as a 145° median axis error;
after the fix the same recording scores ~1°.

When you DO have the headset for ten minutes, bank data for offline work:

```sh
uv run python -m bimanual_teleop.launch.run_teleop --vr orbit --record recordings/session.npz
uv run python scripts/check_roll.py --save recordings/roll.npz   # guided roll capture
```

Every recorded session replays deterministically through the full engine forever
after, so mapping/IK/clutch work never has to wait on hardware again.

## Unity Renderer

Python publishes `render.state` over ZMQ at `tcp://127.0.0.1:8101` and the same
payload as newline-delimited JSON over plain TCP at `127.0.0.1:8102` by default.
The included Unity scaffold at `unity/TeleopRenderer` uses the plain TCP JSON path
so it does not need NetMQ or msgpack packages. It draws both the robot arms and the
operator torso-to-wrist vectors that drive them. It also draws the commanded EE
target separately from the achieved EE marker, so a body-relative mapping problem
or IK lag is visible in the scene. The schema and integration notes are in
`docs/UNITY_BRIDGE.md`.

`scripts/run_unity_validation.py --require` opens the Unity project in batch mode,
compiles the C# scripts, parses a representative `render.state` payload with
`JsonUtility`, validates scene bootstrap/client wiring, and applies the payload to
the primitive arm, hand, and operator-vector renderers. The runner also requires
the Unity log to contain the editor validation success marker, so a zero Unity
process exit alone is not enough. It is skipped by the default hardware-free gate
because this machine may not have Unity installed; `verify_stack.py --unity-editor`
enables it explicitly. The representative payload lives at
`unity/TeleopRenderer/Assets/Editor/render_state_sample.json`, and the Python
static contract check validates that fixture against the current schema and
publisher output. Regenerate it after render payload changes with:

```sh
uv run python scripts/update_unity_fixture.py --write
uv run python scripts/update_unity_fixture.py --check
```

The old local MuJoCo viewer was removed from the runtime. The MJCF assets remain
under `src/bimanual_teleop/sim/models/yam_real/` as measured source geometry for
the programmatic Pinocchio model.

## Motion Mapping (Calibration-Free)

There is no startup calibration ritual. Put the headset on, get tracked, and the
arms follow under one mental model:

> The robot's chest is your torso. WHERE your hand is relative to your torso is
> where the robot's hand goes relative to its chest (absolute, 1:1). HOW your
> hand rotates relative to your body axes is how the EE rotates about the robot
> world axes (relative to clutch-engage).

Position: arm control sees the torso-to-wrist vector `[right, up, forward]`
(headset pose + `vr.torso_from_head`) and targets the robot chest anchor plus that
vector (`mapping.position_mode: absolute`) — hands held in front of YOU are hands
in front of IT, lifting your hand lifts the arm, and walking/turning your body
does nothing. On (re)engage the arm glides onto correspondence over
`mapping.engage_blend_s` seconds instead of snapping. Targets beyond the YAM's
reach (e.g. far above its shoulder-mounted bases) are clamped — the arm tracks
your height as far as its geometry allows. Rotation: the wrist rotation since
clutch-engage is conjugated by the same body→world axis map and applied to the
anchored EE orientation in the world frame — roll→roll, pitch→pitch, yaw→yaw from
ANY starting pose. The previous design inferred a hand↔EE axis correspondence from
a 5-second arms-at-sides hold; done imperfectly (always), it scrambled every
rotation axis (~145° median axis error on a real session — see
`scripts/analyze_session.py`). `vr.calib_seconds` now defaults to 0; setting it >0
re-enables the legacy stillness hold, which only steers arm motion when
`vr.body_relative` is explicitly disabled for diagnostics.

The arm mapping path is:

- `vr/calibrate.py`: head-derived operator body axes + body-relative wrist samples.
- `vr.body_relative`: when true, arm IK receives wrist poses relative to the current
  head/body frame; finger retargeting still receives the raw landmarks.
- `vr.torso_from_head`: the body-frame offset from headset to torso/shoulder proxy.
- `vr/frames.py`: relative/clutch SE(3) mapper (one `R` maps both dp and dR).
- `arms/arm_control.py`: workspace limits, anti-cross guard, smoothing, IK target.
- `arms/ik.py`: two-stage pink IK, wrist position with j1-j3 and hand orientation
  with j4-j6 so a tool-axis roll lands on j6.

Tune physical values in `config/rig.yaml`, especially `vr.torso_from_head`,
`mapping.pos_scale`, workspace limits, and soft joint limits. Leave
`mapping.r_base_from_vr_euler` at zero in the default body-relative mode unless
you are deliberately running legacy diagnostics.

## Hardware

Real YAM control needs Linux/SocketCAN. The hardware entrypoint reuses the same
engine and controllers:

```sh
python -m bimanual_teleop.launch.run_hw --vr orbit
```

Bring up CAN first, one channel per arm:

```sh
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
```

`run_hw` starts idle behind the configured clutch. Keep the YAM motor watchdog and
an external e-stop in the loop for hardware sessions.

For hardware debugging, record the headset/engage stream while running the robot,
then replay it deliberately:

```sh
python -m bimanual_teleop.launch.run_hw --vr orbit --record recordings/hw_session.npz
python -m bimanual_teleop.launch.run_hw --vr replay recordings/hw_session.npz --clutch recorded
```

## Repo Layout

```text
config/rig.yaml              physical, mapping, IK, safety params
docs/UNITY_BRIDGE.md         Unity subscriber schema and coordinate notes
unity/TeleopRenderer/        minimal Unity primitive-arm renderer
scripts/run_synthetic.py     headset-free IK verifier
scripts/check_body_relative.py torso-relative mapping probe
scripts/check_body_relative_render.py Unity render-payload command probe
scripts/check_rig_contract.py default body-relative rig contract verifier
scripts/check_no_mujoco_runtime.py runtime import/dependency guard
scripts/check_yam_geometry.py source-MJCF to Pinocchio geometry verifier
scripts/render_monitor.py    render.state subscriber/debugger
scripts/run_unity_validation.py optional Unity Editor batch validator
scripts/update_unity_fixture.py regenerate/check Unity render-state fixture
src/bimanual_teleop/
  engine.py                  VRFrame + engage -> sink commands
  render_sink.py             Unity render-state publisher
  hardware.py                real robot sink
  vr/                        pose sources, calibration, frame mapping, replay
  arms/                      Pinocchio YAM model, pink IK, arm controller
  hands/                     WebXR/ORBIT landmark retargeting to ORCA joints
  safety/                    clutch, staleness, e-stop state
  bus/                       msgpack ZMQ latest-value transport
tests/                       hardware-free tests
```

## Status

Current verification: `uv run python scripts/verify_stack.py` runs the
hardware-free acceptance gate: Python tests, the torso-relative mapping probe, the
Unity render-payload body-relative command probe, default rig contract, no-MuJoCo
runtime import/dependency guard, YAM source-geometry provenance, synthetic YAM
trajectories, the static Unity render contract plus fixture freshness, launch CLI parsing,
headless teleop smoke, record/replay smoke, and a Unity TCP JSON monitor smoke
that requires an observed active-command bimanual frame. The pytest suite covers
frame mapping, calibration assumptions, IK behavior, replay, safety, ZMQ, ORBIT
Unity-to-WebXR conversion, and end-to-end render ticks. `scripts/run_synthetic.py`
passes line, circle, roll, pitch, and yaw trajectories on both arms, including the
pure-roll check that must land on j6.

Unity Editor rendering can be validated with `scripts/run_unity_validation.py` on a
machine with Unity installed; Quest/in-headset visual validation still requires the
target device.
