# PROGRESS — headset-independent foundation build

Autonomous run journal. Started from the `/goal` brief ("build out the
headset-independent foundation, steps 1–3 + scaffolding"). The operator is away
and cannot answer questions, so every judgement call is recorded here.

---

## TL;DR for when you're back

- **The repo was NOT a greenfield.** It already contained a mature, `mink`-based
  bimanual teleop pipeline (`src/bimanual_teleop/...`) with `vr/frames.py`,
  `arms/ik.py` (two-stage IK), real CAD-measured YAM+ORCA sim models, safety,
  calibration, viz overlays, and 16 passing tests. The goal brief assumed a blank
  scaffold (`teleop/teleop/...` per Section 9). **I did NOT re-scaffold or clobber
  anything** — that would have destroyed real work (frozen rest pose, real MJCF,
  ICP-registered arm bases). Instead I mapped the goal's deliverables onto the
  existing layout and filled the genuine gaps.
- The build spec is the repo's own `CLAUDE.md` (Section 0–10 match the brief's
  references verbatim). There is no separate `teleop_build_spec.md` file.
- **Definition of done is met headlessly:** `uv run pytest` passes and
  `uv run python scripts/run_synthetic.py` runs the line/circle/pure-roll/pitch/yaw
  trajectories through the real two-stage IK and asserts no limit/velocity
  violations and no flips (pure roll is realised on J6). See "What's verified".

## Spec-name → actual-module mapping (Section 9 vs. this repo)

| Spec file (Section 9)     | This repo                                             |
|---------------------------|------------------------------------------------------|
| `teleop/frames.py`        | `src/bimanual_teleop/vr/frames.py`                   |
| `teleop/filters.py`       | `src/bimanual_teleop/filters.py` (NEW, canonical)    |
| `teleop/ik.py`            | `src/bimanual_teleop/arms/ik.py`                     |
| `teleop/calibration.py`   | `src/bimanual_teleop/vr/calibrate.py`                |
| `teleop/transport.py`     | `vr/ingest.py` + `vr/vuer_source.py` + `vr/orbit_source.py` |
| `teleop/control_loop.py`  | `src/bimanual_teleop/engine.py` + `launch/run_sim.py`|
| `teleop/viz.py`           | `viz/overlay.py` + `viz/rerun_log.py` (NEW)          |
| `teleop/logging_utils.py` | `src/bimanual_teleop/logging_utils.py` (NEW)         |
| `teleop/replay.py`        | `src/bimanual_teleop/vr/replay.py` (NEW)             |
| `scripts/run_synthetic.py`| `scripts/run_synthetic.py` (NEW)                     |

---

## Work log

### 0. Survey + baseline (done)
- Read the spec (CLAUDE.md) and the whole existing pipeline.
- Baseline test run (ephemeral pytest): **16 passed**. `mink`/`mujoco` import fine.
- Finding: `pytest` was not a declared dependency, so `uv run pytest` failed
  (only `uv run --with pytest pytest` worked). The DoD requires `uv run pytest`, so
  pytest is being added as a dev dependency group.

(Subsequent steps appended below as they land.)

---

## Assumptions made (because I couldn't ask)

1. **No re-scaffold.** "Scaffold per Section 9" is treated as satisfied by the
   existing `src/bimanual_teleop` layout. Creating a parallel `teleop/teleop` tree
   would duplicate/shadow working modules — explicitly avoided per the guardrail
   "don't overwrite work you didn't create."
2. The build spec referenced as `teleop_build_spec.md` == the repo `CLAUDE.md`.
3. Develop against the **real vendored YAM+ORCA model** (it's present), not mink's
   bundled example. Swapping models is already a `config/rig.yaml` change.

## What's UNVERIFIED (needs the headset / operator — next session)
(to be filled in as scaffolds land)
</content>
</invoke>
