# bimanual-teleop

VR teleoperation for a **torso humanoid**: two **i2rt YAM** 6-DoF arms with an
**ORCA hand** on each, driven by **Meta Quest 3 / 3S** hand-tracking. Your arms
move the robot's arms (Cartesian IK); your fingers move its fingers (retargeting).

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

On the Quest, open `https://<YOUR-PC-LAN-IP>:8012` in the browser, accept the cert,
enter immersive mode, and raise your hands. (Or set `vr.ngrok: true` in `rig.yaml`
to tunnel instead of using a cert.) The arms follow your wrists; your fingers drive
the ORCA hands.

## Tuning (do this once for your rig)

Everything physical lives in **`config/rig.yaml`** — placeholders chosen for a
plausible layout. Dial them to your real aluminium frame:

- **Mounts / rest pose**: `arms.*.mount_pos`, `mount_euler`, `hand_euler`,
  `neutral_q`. Eyeball with `sim_world.py --snap` until the arms+hands sit right.
- **Frame alignment** (the #1 teleop bug — "I move +X, the robot moves wrong"):
  run `uv run python -m bimanual_teleop.tools.frame_check --side left` to see how
  each base-frame axis maps to the world, then set `mapping.r_base_from_vr_euler`
  (mirrored per arm) and `mapping.abd_mirror` (finger spread direction).
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
  sim/   model.py            mjSpec composition (torso + 2 YAM + 2 ORCA)
         sim_world.py        MjModel/MjData, apply commands, viewer/snapshots
         models/yam/         vendored YAM MJCF + meshes (i2rt, MIT)
  launch/ run_sim.py run_hw.py
  tools/  frame_check.py
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

## Provenance / licenses

- YAM model under `sim/models/yam/` is vendored from
  [i2rt-robotics/i2rt](https://github.com/i2rt-robotics/i2rt) (MIT), trimmed to the
  meshes the MJCF uses, with an EE site + position actuators added (see header).
- ORCA hand models are referenced from the sibling `orca_sim` package.
- Retarget math + One-Euro filter ported from the sibling `orca-teleop`.
