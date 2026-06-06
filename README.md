# bimanual-teleop

VR teleoperation for a **torso humanoid**: two **i2rt YAM** arms (6-DoF — the 6th
joint is a real wrist-roll motor at the flange, carrying the **ORCA hand**) on an
AgileX Ranger stand, driven by **Meta
Quest 3 / 3S** hand-tracking. Your arms move the robot's arms (Cartesian IK); your
fingers move its fingers (retargeting).

**Sim-first**: the whole pipeline runs in MuJoCo on macOS with no robot and no
headset. The same engine drives real hardware on a Linux host by swapping one
*sink* — the YAM CAN loop is Linux/SocketCAN-only, so macOS is dev + sim + VR
ingest, and a Linux box runs the arms.

```
Quest 3 (WebXR hand-tracking)                 ┌─────────── one TeleopEngine ───────────┐
  wrists (6-DoF) ─────────────► ArmController ─► mink IK (per-arm, standalone YAM) ─► q[6]
  25 finger joints ──────────► HandController ─► geometric retarget ──────────────► {joint: deg}
                                    ▲                                  │
                                 Supervisor (clutch/deadman,           ▼
                                  staleness, e-stop, states)     sink.set_arm / set_hand
                                                                  ├─ SimWorld  (MuJoCo, now)
                                                                  └─ HardwareSink (YAM CAN + ORCA serial, Linux)
```

## Install

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The `orca_core` and
`orca_sim` repos must be siblings of this folder (editable path deps; the ORCA
hand MuJoCo models are referenced from `orca_sim`, not vendored).

```sh
Developer/
├── orca_core/        # ORCA hand control lib + 17-joint configs
├── orca_sim/         # ORCA MuJoCo hand models (referenced in place)
└── bimanual-teleop/
cd bimanual-teleop
uv sync                 # core sim deps (mujoco, mink, numpy, pyzmq, ...)
uv sync --extra vr      # + Vuer, for real Quest 3 WebXR ingest
```

## Quickstart (sim, no hardware)

```sh
# Headless: render the full pipeline (fake operator) to a GIF — no window needed
uv run python -m bimanual_teleop.launch.run_sim --gif out.gif

# Interactive viewer (macOS needs mjpython, which ships with mujoco):
uv run mjpython -m bimanual_teleop.launch.run_sim            # fake operator
uv run mjpython -m bimanual_teleop.sim.sim_world --demo      # just the robot, sweeping

# One offscreen snapshot of the rig at neutral:
uv run python -m bimanual_teleop.sim.sim_world --snap neutral.png

# Tests (fast, hardware-free):
uv run python tests/test_pipeline.py
```

## Drive it with a real Quest 3 / 3S

WebXR needs HTTPS. Easiest local cert with [mkcert](https://github.com/FiloSottile/mkcert):

```sh
brew install mkcert && mkcert -install
mkcert <YOUR-PC-LAN-IP>          # writes cert.pem / key.pem (point rig.yaml at them)
uv run mjpython -m bimanual_teleop.launch.run_sim --vr vuer
```

On the Quest, open `https://<YOUR-PC-LAN-IP>:8012`, accept the self-signed cert,
enter immersive mode, and raise your hands. The arms follow your wrists; your
fingers drive the ORCA hands.

**On an isolated network (eduroam / campus / most managed Wi-Fi) the LAN route
will NOT work** — the access point blocks device-to-device traffic. Use the public
tunnel instead (no account, works on any network incl. cellular):

```sh
brew install cloudflared
uv run mjpython -m bimanual_teleop.launch.run_sim --tunnel
# wait ~5 s for the banner, then open the printed https://<...>.trycloudflare.com
# URL on the Quest. (Fallback: run scripts/vr_tunnel.sh in a 2nd terminal.)
```

## Tuning (do this once for your rig)

Everything physical lives in **`config/rig.yaml`** — placeholders chosen for a
plausible layout. Dial them to your real aluminium frame:

- **Mounts / rest pose**: `arms.*.mount_pos`, `mount_euler`, `hand_euler`,
  `neutral_q`. Eyeball with `sim_world.py --snap` until the arms+hands sit right.
- **Frame alignment** (the #1 teleop bug — "I move +X, the robot moves wrong"):
  the mapping is wrist-pose → EE target (NOT joint-to-joint), so what bites is
  *frame convention*, not kinematics. Two tools:
  - `uv run python -m bimanual_teleop.tools.frame_check --side left` — static
    snapshots of where each base-frame axis lands in the world.
  - `uv run mjpython -m bimanual_teleop.tools.mapping_studio` — **live side-by-side
    studio**: your hand (25-joint skeleton + wrist triad) floats beside the robot
    arm; SOLID frame/arrow = the robot's actual EE, FAINT = what you're commanding.
    Tune `r_base_from_vr` (I/K pitch, J/L yaw, U/O roll) and `pos_scale` (`-`/`=`)
    with keys and watch faint and solid converge. Runs on synthetic motion with no
    headset; add `--vr vuer` for your real Quest, `--gif out.gif` for a headless clip.
  Then write the values you settle on into `mapping.r_base_from_vr_euler` (mirrored
  per arm) and `mapping.abd_mirror` (finger spread direction).
- **Feel**: `mapping.pos_scale`, `ik.*` (costs/limits), One-Euro smoothing in
  `hands/retarget_core.py`.
- **Clutch / deadman**: hand-tracking has no buttons, so "follow me" is an explicit
  held intent — see `safety/clutch.py` (`AlwaysOn` for sim, `GestureClutch` /
  `KeyboardClutch`, or wire a bluetooth foot pedal to the same interface).

## Hardware day (Linux control host)

The YAM SDK is **Linux + SocketCAN only** (no macOS). Put the arm loop on a Linux
box (NUC / Pi 5 / Jetson) with a USB-CAN adapter per arm; the ORCA hands run over
USB-serial on the same host.

```sh
# 1) CAN up (1 Mbit/s), one channel per arm
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
# 2) i2rt SDK (Ubuntu 22.04, Python 3.11)
git clone https://github.com/i2rt-robotics/i2rt && uv pip install -e i2rt
# 3) ORCA hands: tension + calibrate per orca_core docs, then:
python -m bimanual_teleop.launch.run_hw --vr vuer
```

Mechanical: build each YAM with `gripper_type="no_gripper"` so the flange is free
for the ORCA hand. `run_hw` starts in IDLE (not following); engage via the clutch;
Ctrl+C / e-stop releases torque. Keep the YAM's 400 ms motor watchdog enabled (the
hardware-level deadman). **Validate frames + limits in sim before any operator
session.** For production, split each arm into its own ~250 Hz CAN process,
decoupled from vision/IK via the latest-value bus (`bus/`).

## Repo layout

```
config/rig.yaml              physical + control params (TUNE here)
src/bimanual_teleop/
  config.py                  rig loader
  engine.py                  TeleopEngine: VRFrame + engage → set_arm/set_hand
  hardware.py                HardwareSink (real robot backend)
  vr/    frames.py           VRFrame + relative/clutch SE(3) mapper
         ingest.py           VRSource + FakeVRSource (synthetic operator)
         vuer_source.py      real Quest 3 WebXR ingest
  arms/  ik.py               per-arm mink diff-IK (standalone YAM model)
         arm_control.py      wrist → EE target → IK (+ One-Euro, workspace box)
         yam_driver.py       real YAM over CAN (Linux; import-guarded)
  hands/ retarget_core.py    ported pure retarget + One-Euro (from orca-teleop)
         quest_retarget.py   WebXR 25 joints → ORCA degrees
         hand_control.py     per-hand controller
         joint_map.py        ORCA↔sim joint names, actuator parsing
         real_driver.py      real ORCA hand (orca_core)
  safety/ states.py clutch.py supervisor.py
  sim/   model.py            mjSpec composition (AgileX stand + 2 YAM + 2 ORCA); arm_xml() shared with IK
         sim_world.py        MjModel/MjData, apply commands, viewer/snapshots
         models/yam_real/    vendored 6-DoF YAM arms + AgileX stand (CAD-measured)
  viz/   overlay.py           shared MjvScene primitives (triad/arrow/skeleton)
  launch/ run_sim.py run_hw.py
  tools/  frame_check.py       static base-axis → world snapshots
          mapping_viz.py       scripted-gesture GIF + per-phase error table
          mapping_studio.py    live operator↔robot side-by-side + frame tuning knobs
  bus/   topics.py zmq_io.py  latest-value PUB/SUB for the multi-process path
tests/test_pipeline.py
```

## Status

**Working + tested in sim:** model composition, per-arm IK following, Quest→ORCA
finger retarget, fake-VR end-to-end, safety supervisor (clutch/staleness/e-stop),
ZMQ latest-value bus, hardware driver seam.

**Next:** verify with a real Quest 3 (Vuer payload keys/coordinate frames may need
a small tweak — use `--debug`); tune mounts/frames to the real chassis; split the
hardware path into per-arm ~250 Hz CAN processes over `bus/` for production rates.

## The robot model (6-DoF, real geometry)

The arms are **6-DoF** — the YAM's wrist-roll joint (j6) is a real motor on link5's
circular flange (axis = the mesh-measured flange normal from the friend's CAD); the
ORCA hand bolts onto link6, after that joint. The arm meshes, the **AgileX Ranger
stand**, the per-arm base poses (ICP-registered to CAD, RMS ~1.3 mm), the
flange→hand transforms, and the face-forward home pose all come from
[kairussenberger/Orca-Yam-teleop](https://github.com/kairussenberger/Orca-Yam-teleop)
and are vendored under `sim/models/yam_real/`. Teleop runs a **two-stage diff-IK**
per arm: stage 1 places the wrist with j1–j3, stage 2 orients the end-effector with
j4–j6 (so a wrist twist lands on j6 rather than swinging the forearm). The one thing
still to calibrate before hardware is `mapping.r_base_from_vr_euler` (run the
`tools/mapping_studio` frame tuner).

## Provenance / licenses

- 6-DoF YAM arm model + AgileX stand under `sim/models/yam_real/`: from
  [kairussenberger/Orca-Yam-teleop](https://github.com/kairussenberger/Orca-Yam-teleop)
  (the real rig's MuJoCo setup), itself derived from the i2rt YAM URDF.
- ORCA hand models are referenced from the sibling `orca_sim` package.
- Retarget math + One-Euro filter ported from the sibling `orca-teleop`.
