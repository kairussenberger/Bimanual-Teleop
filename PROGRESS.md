# PROGRESS — headset-independent foundation build

Autonomous run journal (operator away ~1h, could not be asked). Every judgement
call is recorded here. Work was done on branch **`auto/teleop-foundation`** (not
`main`) so it's easy to review/merge/revert.

---

## TL;DR for when you're back

- **The repo was NOT greenfield.** It already had a mature, `mink`-based bimanual
  teleop pipeline under `src/bimanual_teleop/` (frames, two-stage IK, real CAD
  YAM+ORCA sim models, safety, calibration, viz, 16 passing tests). The goal brief
  assumed a blank `teleop/teleop/` scaffold (Section 9). **I did not re-scaffold or
  clobber anything** — that would have destroyed real work (frozen rest pose, real
  MJCF, ICP-registered bases). I mapped the goal's deliverables onto the existing
  layout and filled the genuine gaps.
- The build spec is the repo's own `CLAUDE.md` (its Section 0–10 match the brief's
  references verbatim). There is no separate `teleop_build_spec.md`.
- **Definition of done is met (headlessly):**
  - `uv run pytest` → **52 passed** (was 16; +36 new).
  - `uv run python scripts/run_synthetic.py` → **ALL trajectories PASS** on both
    arms (line, circle, pure roll/pitch/yaw): pose error ≤0.72 cm / ≤0°, joint
    velocity ≤1.6 rad/s (limit 12), no flips, within soft limits, and **pure roll
    lands on j6**. Writes `out/run_synthetic.gif` overlaying achieved (solid) vs
    commanded (faint) EE triads at each hand.

### Headline technical finding
The two-stage IK realises a **pure tool-axis roll on j6** to <2° with j1–j5 barely
moving (tests/test_ik.py + the synthetic harness). So **the IK is sound** — if a
real wrist roll still fails with the headset on, the bug is in **frames/tracking
(R_align / calibration)**, NOT in IK or joint limits. That's exactly the isolation
the spec's Section 7 asks for, now provable on demand.

## Spec-name → actual-module mapping (Section 9 vs. this repo)

| Spec file (Section 9)     | This repo                                             | Status |
|---------------------------|------------------------------------------------------|--------|
| `teleop/frames.py`        | `src/bimanual_teleop/vr/frames.py`                   | extended + tested |
| `teleop/filters.py`       | `src/bimanual_teleop/filters.py` (NEW, canonical)    | new + tested |
| `teleop/ik.py`            | `src/bimanual_teleop/arms/ik.py`                     | extended + tested |
| `teleop/calibration.py`   | `src/bimanual_teleop/vr/calibrate.py`               | pre-existing (UNVERIFIED w/ headset) |
| `teleop/transport.py`     | `vr/ingest.py` + `vr/vuer_source.py` + `vr/orbit_source.py` | pre-existing |
| `teleop/control_loop.py`  | `src/bimanual_teleop/engine.py` + `launch/run_sim.py`| pre-existing |
| `teleop/viz.py`           | `viz/overlay.py` + `viz/rerun_log.py` (NEW)          | overlay pre-existing; rerun new |
| `teleop/logging_utils.py` | `src/bimanual_teleop/logging_utils.py` (NEW)         | new + tested |
| `teleop/replay.py`        | `src/bimanual_teleop/vr/replay.py` (NEW)             | new + tested (live capture UNVERIFIED) |
| `scripts/run_synthetic.py`| `scripts/run_synthetic.py` (NEW)                     | new + tested |

---

## Work log (commits on `auto/teleop-foundation`)

1. **Dev tooling + journal.** Added `pytest` as a dev dependency group (so the DoD
   command `uv run pytest` works — it previously failed: pytest wasn't a dep) and
   `rerun-sdk` as an optional `telemetry` extra. Committed the spec (`CLAUDE.md`)
   and this journal.
2. **frames: R_align + change-of-basis (tests first).** Added the Section 3
   primitives explicitly — `conjugate_rotation(B,dR)=B·dR·Bᵀ` and the quaternion
   form `change_basis_quat(q,dq)=q·dq·q⁻¹`, plus `R_to_quat`/`quat_mul`/`quat_conj`/
   `quat_inv`/`quat_from_axis_angle` and an `R_ALIGN` alias for `WEBXR_TO_WORLD`.
   `tests/test_frames.py` (written first) pins the **+90° roll** invariant: a roll
   about the hand forward axis → a roll about the robot tool axis and *nothing else*,
   both via the helper and end-to-end through `ClutchMapper`. The existing relative
   mapping already satisfied it.
3. **filters: canonical One-Euro.** The proven webcam-ported One-Euro lived inside
   `hands/retarget_core`; moved it verbatim to `filters.py` (single source, spec's
   `filters.py`), re-exported for back-compat, added `OneEuroVecFilter`. Behaviour
   unchanged (same defaults). `tests/test_filters.py` pins passthrough/no-DC-offset/
   no-overshoot/higher-beta-less-lag/jitter-attenuation/vec==dict/zero-dt-safety.
4. **ik: elbow limit + observability + collision hook.** Documented invariant #5
   (model `jnt_range` + `ConfigurationLimit` + soft limits cap elbow j3
   hyperextension). Added `limit_margins()`/`within_limits()` for the HUD/tests and
   an opt-in `collision_pairs=` hook (off by default; standalone arm geoms are
   visual-only). `tests/test_ik.py` adds the **J6 isolation test**.
5. **observability: logging_utils + Rerun.** `get_logger` (levelled), `RateMeter`
   (EWMA loop Hz), `TelemetryRing` (latest-wins ring + CSV). `viz/rerun_log.py`:
   optional Rerun dashboard (3D transforms, triads, scalars, text), guarded to a
   no-op when the dep is absent. Tested.
6. **synthetic harness (the headline).** `scripts/run_synthetic.py` drives the
   two-stage IK with scripted EE targets (line/circle/pure roll/pitch/yaw), ease-in
   so targets never teleport. Headless verify + PASS/FAIL table + best-effort GIF
   (achieved vs commanded triads) + optional CSV/`--rerun`/`--view`. Exit 0 iff all
   pass. `tests/test_synthetic.py` smoke-tests it.
7. **replay scaffold.** `vr/replay.py`: `SessionRecorder` + `ReplaySource` (drop-in
   VRSource), `.npz` format, `replay` transport in `make_source`. Tested round-trip.

---

## Assumptions made (because I couldn't ask)

1. **No re-scaffold.** "Scaffold per Section 9" is treated as satisfied by the
   existing `src/bimanual_teleop` layout; a parallel `teleop/teleop` tree would
   duplicate/shadow working modules (guardrail: don't overwrite work you didn't
   create). Mapping table above is the bridge.
2. `teleop_build_spec.md` (referenced by the brief) == the repo `CLAUDE.md`.
3. Develop against the **real vendored YAM+ORCA model** (it's present and is the
   IK/sim source of truth via `sim.model.arm_xml`), not mink's bundled example.
   Swapping models is a `config/rig.yaml` change (the brief's "config change only").
4. Synthetic mode drives **EE targets relative to home** (the spec's Section 7
   intent) — this isolates IK and is NOT absolute teleop mapping; the teleop input
   path remains relative+clutch (`ClutchMapper`, untouched).
5. **Default `run_synthetic` mode is headless** (verify + GIF), because a live
   MuJoCo window can't be driven in this autonomous run and would hang. `--view`
   (mjpython) is the live window for you.
6. Committed on a branch, not `main`, since this was an unattended batch of changes.
7. `j3` is the elbow (soft limit already caps its hyperextension); the harness +
   tests assert all joints (incl. j3) stay within soft limits.

---

## Verified this run ✅
- `uv run pytest` → 52 passed (frames incl. +90° roll, One-Euro, IK incl. J6
  isolation + elbow/limit enforcement, observability, synthetic, replay, and all
  pre-existing pipeline tests).
- `uv run python scripts/run_synthetic.py` → all trajectories PASS both arms; GIF +
  CSV artifacts produced; pure roll realised on j6.
- One-Euro refactor is behaviour-preserving (existing finger/arm tests still green).
- Rerun logger degrades to a silent no-op without `rerun-sdk` (it's not installed).

## UNVERIFIED — needs the headset / operator (next session) ⚠️
These are correct-by-construction or unit-tested in their pure parts, but their
real-hardware/operator path could not run without the Quest:
1. **`R_align` real-wrist validation.** The static `R_align = WEBXR_TO_WORLD` is
   correct on synthetic data and unit-tested. The *per-session yaw correction* and
   the hand→tool correspondence `P` come from `vr/calibrate.py` driven by a real
   resting-stance capture — these need a headset to validate. Run the spec's Section
   8 step 5: with the headset on, a real wrist roll should reproduce the synthetic
   roll result (faint/solid triads spinning together about blue in `run_sim`).
2. **Live VR transports** (`vuer_source.py`, `orbit_source.py`): their pure parsing
   logic is unit-tested (e.g. `test_orbit_source_unity_to_webxr`), but end-to-end
   streaming from a Quest is unverified here.
3. **Live session recording** (`SessionRecorder` fed by a real source). The
   record→save→load→replay machinery is tested on synthetic frames; capturing a
   real session is the only unverified link.
4. **Calibration state machine** end-to-end with an operator holding the stance.

## Recommended next steps (in spec bring-up order)
- **Step 4–5 (headset):** wear the Quest, run `uv run mjpython -m
  bimanual_teleop.launch.run_sim --vr vuer`, do the resting-stance calibration, and
  confirm a real wrist roll spins the commanded+achieved triads together about the
  tool axis. If they diverge, it's `R_align`/`P` (frames/calibration) — the IK is
  already cleared by the synthetic roll test.
- Record a short session (wire `SessionRecorder` into `run_sim`), then bisect feel
  changes with `--vr replay` instead of re-wearing the headset.
- If you want true cross-arm **self-collision avoidance**, it needs a combined
  collidable model (the per-arm standalone IK can't see the other arm); enable geoms
  + pass `collision_pairs=` to `ArmIK` (hook is in place). Today it's mitigated by
  the anti-cross world-Y guard + workspace box + soft limits.
- Optional: `uv sync --extra telemetry` then `run_synthetic.py --rerun` for the live
  Rerun dashboard (3D frames + error/vel/margin time-series).

## How to verify (copy/paste)
```
uv run pytest -q                                   # 52 passed
uv run python scripts/run_synthetic.py             # table + out/run_synthetic.gif
uv run python scripts/run_synthetic.py --no-gif    # fastest verify only
uv run mjpython scripts/run_synthetic.py --view     # live window (when you're here)
```

## Adversarial verification (multi-agent workflow)
Ran a 5-agent verification workflow (4 adversarial lenses + synthesis) over the
branch diff. **Verdict: ship_with_notes — no must-fix items, no invariant
violations, no broken tests, 52/52 re-verified independently.** It confirmed:
relative+clutch mapping (not absolute), mink-only IK (no pseudoinverse anywhere),
limits enforced in model + IK + elbow soft cap, frame transforms tested. It
re-checked and *rejected* its own two "high" flags (the +90° roll test does NOT
pass with an identity R_ALIGN; the j6 axis genuinely equals the EE tool axis).

Addressed the high-value notes in commit "harden: …": de-circularized + tightened
the J6 test, full-matrix ground truth on the +90° roll test, attenuation assertion
on the filter step test, wrist-rotation round-trip in replay, loop-rate↔IK-dt
consistency + flip-threshold fix in the harness, explicit replay_path error,
unified CSV float formatting, and documented the inert `abs_orientation` flag.

Deferred (genuinely low-priority, noted for later):
- Independently verify the elbow invariant by commanding a hyperextension target and
  asserting EE error grows (today's check confirms the imposed soft limit holds).
- Add a large-roll (>j6 soft cap) trajectory to exercise the saturation/regrip
  regime (current ROLL_AMP≈29° stays inside the cap).
- Either remove `abs_orientation` entirely (touches config + arm_control) or wire an
  absolute branch with a differentiating test — documented as inert for now.
- The Section-3 quaternion change-of-basis helpers in `frames.py` are reference/
  test-only; the live arm path maps orientation via `ClutchMapper.set_P`. Keep them
  in sync or route the live path through them later.
- The record half of record/replay (`SessionRecorder`) has no production caller yet;
  wire it into `run_sim` (and feed `engaged_at` to the engine) when debugging replay.

## Known limitations / notes
- Self-collision avoidance is a documented hook, not active (see above).
- `out/`, `*.gif`, telemetry dumps are gitignored (generated artifacts).
- The first journal commit accidentally included a couple of stray editor tags in
  this file; cleaned in the working tree (and this rewrite). No code was affected.
