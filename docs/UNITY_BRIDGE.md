# Unity Bridge

Python publishes the robot render state in two forms:

- ZMQ/msgpack `render.state` for Python tools and high-rate process wiring.
- Plain TCP newline-delimited JSON for Unity, so the Unity project can use only
  `TcpClient` and `JsonUtility`.

The included Unity scaffold in `unity/TeleopRenderer` uses the TCP JSON path.

## Transport

- ZMQ endpoint: `tcp://127.0.0.1:8101` by default.
- Unity JSON endpoint: `tcp://127.0.0.1:8102` by default.
- Topic: `render.state`.
- ZMQ encoding: multipart ZMQ message, topic string followed by msgpack payload.
- Unity JSON encoding: one compact JSON render-state object per line. Python uses
  strict JSON for this stream; frames containing `NaN` or `Infinity` are dropped
  instead of being sent to Unity.
- USB Quest path: `adb reverse tcp:8101 tcp:8101` and
  `adb reverse tcp:8102 tcp:8102` map Quest localhost to the PC publishers.
  `run_teleop --vr orbit` sets both up automatically when `adb` exists.
- Debug subscribers:
  - ZMQ: `uv run python scripts/render_monitor.py`
  - Unity JSON: `uv run python scripts/render_monitor.py --transport json`
  - Unity JSON with hand schema check:
    `uv run python scripts/render_monitor.py --transport json --require-hand-render --require-bimanual-state --require-command-target --require-frame`

The endpoints can be changed with `vr.render_endpoint` and
`vr.unity_json_endpoint` in `config/rig.yaml`. Both render transports are
best-effort: if one port is already busy, Python logs a warning and teleop
continues with any other render transport that started.

## Message Shape

```text
{
  "v": 2,
  "stamp": float,
  "arms": {
    "left": {
      "q": [6 floats],
      "link_pos": [x0, y0, z0, x1, y1, z1, ...],
      "ee_pos": [x, y, z],
      "ee_quat": [w, x, y, z],
      "cmd_pos": [x, y, z] | null,
      "cmd_quat": [w, x, y, z] | null,
      "margins": [6 floats]
    },
    "right": { ... }
  },
  "hands": {
    "left": {"orca_joint": degrees, ...},
    "right": {"orca_joint": degrees, ...}
  },
  "hand_render": {
    "left": {
      "names": ["wrist", "thumb_cmc", ...],
      "q": [degrees in the same order]
    },
    "right": { ... }
  },
  "op": {
    "torso_from_head": [right, up, forward],
    "head_pos": [x, y, z] | null,
    "torso_pos": [x, y, z] | null,
    "hands": {
      "left": {
        "tracked": bool,
        "wrist_body": [right, up, forward] | null,
        "raw_wrist": [x, y, z] | null
      },
      "right": { ... }
    }
  },
  "status": {
    "engaged": {"left": bool, "right": bool},
    "tracked": {"left": bool, "right": bool},
    "calib": null | {
      "active": bool,
      "phase": "wait" | "hold" | "done",
      "progress": float,
      "remaining": float,
      "left": bool,
      "right": bool,
      "msg": string
    },
    "hz": float
  }
}
```

`link_pos`, `ee_pos`, `ee_quat`, `cmd_pos`, and `cmd_quat` are in robot world
coordinates. `link_pos` is the flattened base, j1..j6, and EE polyline from
Python's live Pinocchio state; the included Unity renderer draws primitive joints
and links from this array instead of duplicating FK constants. `cmd_pos` is the
effective commanded target after body-relative mapping, filtering, workspace
clipping, and anti-cross clamping; Unity draws it separately from the achieved EE
marker and connects the two with an error line, so mapping, clamping, and IK lag
are visible. Quaternions are `w, x, y, z`; Unity APIs usually use `x, y, z, w`, so
reorder before constructing a `Quaternion`.

`hands` is a dynamic `{orca_joint: degrees}` dictionary for Python subscribers.
Unity's built-in `JsonUtility` cannot parse arbitrary dictionaries, so Unity uses
the fixed-shape `hand_render` block: `names[i]` identifies the ORCA joint and
`q[i]` is the corresponding degree value. The included Unity scaffold draws
primitive ORCA palms and fingers from `hand_render` at each arm's achieved EE pose.

## Coordinates

The internal robot world is right-handed. The robot faces world `-X`, world `+Z`
is up, and world `+Y` separates the arms laterally. ORBIT input is Unity style
left-handed with `+Z` forward; `vr/orbit_source.py` converts it to the internal
WebXR-style frame before the teleop engine sees it.

Arm motion is body-relative before it reaches IK: Python converts raw wrist poses
to the current torso/body frame using `vr.torso_from_head`, so whole-body/head
translation does not move the robot arms. The `op.hands.*.wrist_body` fields are
the exact torso-to-wrist vectors used for arm control; the included Unity scaffold
draws them as an operator overlay. In body-relative mode, `status.tracked.*` is true
only when that side has a finite torso-to-wrist vector that can drive arm motion.
If the headset pose is missing, Python publishes
`status.tracked.* = false`, `op.hands.*.tracked = false`, and
`op.hands.*.wrist_body = null` instead of falling back to raw room-space hand
coordinates. Unity also rejects non-finite body/operator values before accepting a
state. Unity receives the resulting robot state in robot world coordinates.

For rendering in Unity, keep one explicit conversion at the boundary and use it
consistently for positions, rotations, and model bind poses. Do not add a second
axis flip in individual joints. The included scaffold keeps that boundary in
`Assets/Scripts/TeleopUnityFrame.cs`; arm, hand, and operator renderers call that
helper instead of owning private conversion methods.

## Included Unity Scaffold

Open `unity/TeleopRenderer` in Unity and press Play. The scene bootstrap creates a
camera, lights, a TCP JSON client, primitive YAM arm renderers, primitive ORCA hand
renderers, the operator torso-to-wrist overlay, and a small status HUD. It connects
to `127.0.0.1:8102` and draws the latest state from Python.
When embedded in an existing Unity scene, the bootstrap reuses an existing camera,
light, and `Floor` object where possible instead of duplicating scene support
objects. If a scene already contains a saved/manual `TeleopRenderClient`, the
runtime bootstrap still ensures those support objects before it returns.

The HUD is intentionally dependency-free and uses Unity's built-in IMGUI path. It
surfaces TCP connection state, schema/stale/error state, loop rate, engagement and
tracking flags, calibration progress, and whether the operator head/wrist vectors
are currently usable. It also prints left/right achieved-to-command EE error in
centimeters, matching the gold error lines in the scene.

The bootstrap configures the Unity process for headset rendering by disabling
vSync frame throttling, targeting 72 FPS, and keeping the device awake while the
renderer is active.

Run Python beside it:

```sh
uv run python -m bimanual_teleop.launch.run_teleop --vr fake
```

On machines with Unity Editor installed, run the Editor-side batch validation from
the repo root:

```sh
uv run python scripts/run_unity_validation.py --require
```

That command opens `unity/TeleopRenderer`, compiles the C# scripts, parses a
representative `render.state` JSON payload with Unity's `JsonUtility`, validates
the scene bootstrap/client wiring, and applies the payload to the primitive arm,
hand, operator-vector, and HUD paths. It also requires the Unity log to contain the
editor validation success marker, so a zero Unity process exit alone is not enough.
The payload fixture is
`unity/TeleopRenderer/Assets/Editor/render_state_sample.json`; the Python static
contract check validates that fixture against the current render schema and
publisher output. After changing `render.state`, regenerate and check it with:

```sh
uv run python scripts/update_unity_fixture.py --write
uv run python scripts/update_unity_fixture.py --check
```

## ZMQ Subscriber Logic

```csharp
// Pseudocode for the ZMQ/msgpack path. The included Unity project uses TCP JSON
// instead, so this is only needed if you want NetMQ/msgpack in Unity.
var sub = new SubscriberSocket();
sub.Connect("tcp://127.0.0.1:8101");
sub.Subscribe("render.state");

while (running) {
    while (sub.TryReceiveMultipartBytes(TimeSpan.Zero, ref frames)) {
        var payload = MessagePackSerializer.Deserialize<RenderState>(frames[1]);
        latest = payload;
    }
    if (latest != null) {
        ApplyArm("left", latest.arms.left.q);
        ApplyArm("right", latest.arms.right.q);
        DrawHud(latest.status);
    }
}
```

Use latest-value behavior on the Unity side too. Dropping old render states is
correct; replaying stale robot states adds visible lag. The included TCP client
keeps only the newest unread JSON line before applying it on the Unity main thread.

## Verification

Before launching Unity, verify Python is publishing:

```sh
uv run python -m bimanual_teleop.launch.run_teleop --vr fake --duration 5
uv run python scripts/render_monitor.py --seconds 5
uv run python scripts/render_monitor.py --transport json --require-hand-render --require-bimanual-state --require-command-target --require-frame --seconds 5
```

For the TCP JSON stream, open `unity/TeleopRenderer` and press Play while
`run_teleop` is running.

For Unity-side compilation and renderer-apply validation on a machine with Unity:

```sh
uv run python scripts/run_unity_validation.py --require
uv run python scripts/verify_stack.py --unity-editor
```

For a real headset, run ORBIT and then:

```sh
uv run python -m bimanual_teleop.launch.run_teleop --vr orbit --clutch gesture
```
