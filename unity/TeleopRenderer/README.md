# TeleopRenderer Unity Project

Minimal Unity-side renderer for `bimanual-teleop`.

It connects to the Python plain-TCP JSON stream at `127.0.0.1:8102`, reads the
latest render state, and draws both YAM arms using primitive joints/links. This is
not a polished robot asset pipeline; it also draws simple ORCA palms/fingers from
ordered joint degrees. It is a dependency-light Unity simulation view that proves
the MuJoCo-free runtime can drive Unity directly.

## Run

1. Open this folder in Unity.
2. Start Play mode.
3. In another terminal from the repo root:

   ```sh
   uv run python -m bimanual_teleop.launch.run_teleop --vr fake
   ```

For a Quest build over USB, make sure the Python launcher has set:

```sh
adb reverse tcp:8102 tcp:8102
```

`run_teleop --vr orbit` does this automatically when `adb` is available.

## What It Draws

- Spheres: YAM joint positions from Python-published Pinocchio link points.
- Cylinders: simple arm links drawn between those published points.
- Bright end sphere: achieved EE position reported by Python FK.
- Gold target sphere: commanded EE target after body-relative mapping, filtering,
  workspace clipping, and anti-cross clamping.
- Gold line: achieved-to-command error, useful for seeing mapping/clamping/IK lag
  in the Unity scene.
- ORCA palms/fingers: primitive hand state from `render.state.hand_render.*`.
- Operator overlay: torso proxy plus left/right torso-to-wrist vectors from
  `render.state.op.hands.*.wrist_body`.
- Status HUD: TCP connection state, schema/stale/error state, loop rate,
  engagement/tracking flags, calibration progress, operator head/wrist status, and
  numeric achieved-to-command EE error.
- Color changes: engaged/tracked status from the teleop state.

The Unity arm renderer does not duplicate FK constants. Python publishes the
flattened base, j1..j6, and EE polyline in `render.state.arms.*.link_pos`, and
Unity only converts those robot-world points into Unity coordinates.
The coordinate boundary is centralized in `Assets/Scripts/TeleopUnityFrame.cs`,
which is shared by the arm, hand, and operator overlay renderers.

The TCP client reconnects automatically while Play mode is running, uses a bounded
connect timeout, and closes its socket when Play mode stops, so restarting Python
or Unity does not require editor restart. The HUD is built with Unity's built-in
IMGUI path, so it does not require UI packages in the scaffold.

The scene bootstrap also configures the renderer process for headset use by
disabling vSync frame throttling, targeting 72 FPS, and preventing device sleep
while the render scene is active.

## Editor Validation

From the repo root, a machine with Unity installed can run the Unity-side validation
in batch mode:

```sh
uv run python scripts/run_unity_validation.py --require
```

This opens this Unity project, compiles the C# scripts, parses a representative
`render.state` JSON payload with `JsonUtility`, validates the scene bootstrap/client
wiring, and applies that state to the arm, hand, and operator-vector renderers. The
same validation is available in the Unity menu at `Teleop/Run Renderer Validation`.
The sample payload is `Assets/Editor/render_state_sample.json` and is also checked
by Python's static Unity contract test.

The validation also runs `TeleopSceneAsset.EnsureRendererScene()`, which saves
`Assets/Scenes/TeleopRenderer.unity` and registers it in Unity build settings. The
same action is available from `Teleop/Ensure Renderer Scene` before making a Quest
or desktop build.
