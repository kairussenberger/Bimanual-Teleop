# Replay library — curated probe tapes (recorded 2026-06-12)

The canonical recordings for the sim→real checklist (ARCHITECTURE.md step 5:
replay on hardware, "no surprises allowed") and for regression-scoring mapping
changes. The tapes are COMMITTED under `replay_library/` (trimmed + session
fit embedded — self-contained on any clone); working copies and the untrimmed
`.raw.npz` originals stay in the gitignored `recordings/`.

Every tape is TRIMMED (the walk-to-the-laptop tail, and pick_place's first
10 s) and EMBEDS the calibration that ran during its session (`calib_json` in
the npz — replay/analyze apply it automatically; identity-scoring these tapes
is meaningless because ORBIT anchors moved ~1 m between sessions, see
SESSION_NOTES 2026-06-12). Untrimmed originals are kept as `<name>.raw.npz`;
each session's fit is also snapshotted as `<name>.calib.json`.

| tape | len | what it probes | analyze (median): abs-corr / ori / IK | verdict |
|------|-----|----------------|----------------------------------------|---------|
| roll_right_left.npz | 42 s | pure wrist rolls to both stops, j6 saturation | 2.0 cm / 4.3° / 0.7° | **PASS** — primary hardware-day tape |
| reach_box.npz | 50 s | workspace envelope sweeps | 2.7 cm / 6.1° / 4.3° | marginal: ori 1.1° over the 5° gate (motion lag during fast reaches) — usable |
| wrist_swing.npz | 62 s | j4/j5 pitch+yaw swings | 5.5 cm / **0.18°** / 0.2° | PASS in substance; "FAIL" is the direction-error gate reading noise on a position-static probe |
| clap.npz | 77 s | separation guard at contact | 12.4 cm / 13.4° / 0.9° | corr ≈ the guard's deliberate push (analyzer can't see pair-separation); CONTAINS the full-clasp crossing misbehavior (open issue) |
| pick_place.npz | 66 s | natural manipulation | 12.0 cm / 16.6° / 1.2° | degraded by grasp-pose tracking noise (see fingers) |
| fingers.npz | 76 s | finger curls/opposition, arms parked | 16.8 cm / **71°** / 0.7° | retargeting dataset; quantifies the curl-pose WRIST tracking noise (the "right wrist moves too much" report) — open issue |

Session fits (operator that day; axis_scale [lat, up, fwd] / body_offset):
roll_right_left [1.376, 0.971, 1.223] / [0, −0.889, 0.001] (retrofitted from
the engine log — lat_center/knots lost, plain lateral scale); reach_box
[1.32, 0.94, 1.426] / [0, −1.506, 0.808]; pick_place [1.399, 0.977, 1.366] /
[0, −1.486, 0.829]; clap [1.295, 1.017, 1.312] / [0, −1.794, 0.781];
wrist_swing [1.329, 0.948, 1.261] / [0, −1.472, 0.127]; fingers
[0.812, 0.903, 1.488] / [0, −1.492, 0.766] (lat 0.81 is an outlier vs
1.29–1.40 elsewhere — wide-spread calibration; harmless for a fingers tape).

Still to record: engage_cycles (gesture clutch), recenter_trip +
occlusion_no_trip (anchor-guard live validation), one deliberately sloppy
calibration (grade-threshold data).

Known issues feeding from these tapes: (1) full-clasp crossing in clap.npz —
separation push direction degenerates at contact; (2) curl-pose wrist noise in
fingers.npz / pick_place.npz — raw wrist attitude swings ~71° median while
fingers curl (keypoint/wrist-stream tracking, not mapping). Analyzer follow-ups
(not yet done): exclude pair-separation pushes from abs-corr, gate direction
error on displacement magnitude, per-probe-class verdict profiles.

Curation commands:

    uv run python scripts/trim_session.py recordings/tape.npz --head S --tail S \
        [--calib recordings/tape.calib.json] [-o out.npz]
    uv run python scripts/analyze_session.py recordings/tape.npz [--no-calib]
