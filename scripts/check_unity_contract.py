#!/usr/bin/env python
"""Static Unity/Python render-contract checks that do not require Unity Editor.

Unity's `JsonUtility` only handles serializable field-based classes, not arbitrary
runtime dictionaries. This script checks that the C# DTOs consume the fixed parts
of Python's `render.state` payload (`arms.left/right`, `status`, and
`op.hands.left/right`) and that the primitive renderers use Python-published
robot/hand state instead of duplicating FK constants.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from bimanual_teleop.bus import topics
from bimanual_teleop.config import SIDES, load_rig
from bimanual_teleop.hands.joint_map import ORCA_JOINT_ORDER
from update_unity_fixture import make_fixture, normalized_text


REPO = Path(__file__).resolve().parents[1]
UNITY = REPO / "unity" / "TeleopRenderer"
SCRIPTS = REPO / "unity" / "TeleopRenderer" / "Assets" / "Scripts"
EDITOR = REPO / "unity" / "TeleopRenderer" / "Assets" / "Editor"
SAMPLE = EDITOR / "render_state_sample.json"
UNITY_RUNNER = REPO / "scripts" / "run_unity_validation.py"
GITIGNORE = REPO / ".gitignore"
REQUIRED_FILES = [
    UNITY / "Packages" / "manifest.json",
    UNITY / "ProjectSettings" / "ProjectVersion.txt",
    SCRIPTS / "TeleopRenderClient.cs",
    SCRIPTS / "TeleopUnityFrame.cs",
    SCRIPTS / "TeleopUnityMaterials.cs",
    SCRIPTS / "TeleopSceneBootstrap.cs",
    SCRIPTS / "TeleopStatusHud.cs",
    SCRIPTS / "YamArmRenderer.cs",
    SCRIPTS / "OrcaHandRenderer.cs",
    SCRIPTS / "OperatorVectorRenderer.cs",
    EDITOR / "TeleopEditorValidation.cs",
    EDITOR / "TeleopSceneAsset.cs",
    SAMPLE,
    UNITY_RUNNER,
    UNITY / "README.md",
    GITIGNORE,
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_comments_and_strings(src: str) -> str:
    out: list[str] = []
    i = 0
    mode = "code"
    while i < len(src):
        c = src[i]
        n = src[i + 1] if i + 1 < len(src) else ""
        if mode == "code":
            if c == "/" and n == "/":
                mode = "line_comment"
                i += 2
                continue
            if c == "/" and n == "*":
                mode = "block_comment"
                i += 2
                continue
            if c == "@" and n == '"':
                mode = "verbatim_string"
                out.append('"')
                i += 2
                continue
            if c == '"':
                mode = "string"
                out.append('"')
                i += 1
                continue
            out.append(c)
        elif mode == "line_comment":
            if c == "\n":
                mode = "code"
                out.append("\n")
        elif mode == "block_comment":
            if c == "*" and n == "/":
                mode = "code"
                i += 2
                continue
            if c == "\n":
                out.append("\n")
        elif mode == "string":
            if c == "\\":
                i += 2
                continue
            if c == '"':
                mode = "code"
                out.append('"')
        elif mode == "verbatim_string":
            if c == '"' and n == '"':
                i += 2
                continue
            if c == '"':
                mode = "code"
                out.append('"')
        i += 1
    if mode in {"block_comment", "string", "verbatim_string"}:
        raise AssertionError("unterminated C# comment/string literal")
    return "".join(out)


def _assert_balanced(path: Path, src: str) -> None:
    pairs = {")": "(", "]": "[", "}": "{"}
    opens = set(pairs.values())
    stack: list[tuple[str, int]] = []
    clean = _strip_comments_and_strings(src)
    for line, text in enumerate(clean.splitlines(), start=1):
        for c in text:
            if c in opens:
                stack.append((c, line))
            elif c in pairs:
                if not stack or stack[-1][0] != pairs[c]:
                    raise AssertionError(f"{path.name}:{line}: unmatched {c}")
                stack.pop()
    if stack:
        c, line = stack[-1]
        raise AssertionError(f"{path.name}:{line}: unmatched {c}")


def _class_body(src: str, class_name: str) -> str:
    m = re.search(rf"\bpublic\s+(?:sealed\s+)?class\s+{re.escape(class_name)}\b[^\{{]*\{{", src)
    if not m:
        raise AssertionError(f"missing C# class {class_name}")
    start = m.end()
    depth = 1
    i = start
    while i < len(src) and depth:
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
        i += 1
    if depth:
        raise AssertionError(f"unterminated C# class {class_name}")
    return src[start:i - 1]


def _fields(src: str, class_name: str) -> set[str]:
    body = _class_body(src, class_name)
    return set(re.findall(r"\bpublic\s+[A-Za-z_][A-Za-z0-9_<>,\[\]]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*;", body))


def _assert_fields(src: str, class_name: str, required: set[str]) -> None:
    got = _fields(src, class_name)
    missing = sorted(required - got)
    if missing:
        raise AssertionError(f"{class_name} missing fields: {', '.join(missing)}")


def _assert_sample_payload(sample: dict) -> None:
    _assert_json_finite(sample)
    if sample.get("v") != topics.SCHEMA_VERSION:
        raise AssertionError("Unity sample render state schema version is stale")
    if set(sample.get("arms", {}).keys()) != set(SIDES):
        raise AssertionError("Unity sample must include both arm sides")
    if set(sample.get("hand_render", {}).keys()) != set(SIDES):
        raise AssertionError("Unity sample must include both hand_render sides")
    if set(sample.get("op", {}).get("hands", {}).keys()) != set(SIDES):
        raise AssertionError("Unity sample must include both operator hand sides")
    for side in SIDES:
        arm = sample["arms"][side]
        for field, n in (("q", 6), ("link_pos", 24), ("ee_pos", 3), ("ee_quat", 4), ("cmd_pos", 3), ("cmd_quat", 4), ("margins", 6)):
            if len(arm.get(field, [])) != n:
                raise AssertionError(f"sample arms.{side}.{field} must have length {n}")
        hand = sample["hand_render"][side]
        if hand.get("names") != ORCA_JOINT_ORDER:
            raise AssertionError(f"sample hand_render.{side}.names must match ORCA_JOINT_ORDER")
        if len(hand.get("q", [])) != len(ORCA_JOINT_ORDER):
            raise AssertionError(f"sample hand_render.{side}.q length mismatch")
        if set(sample.get("hands", {}).get(side, {}).keys()) != set(ORCA_JOINT_ORDER):
            raise AssertionError(f"sample hands.{side} dynamic dict must expose all ORCA joints")
        op_hand = sample["op"]["hands"][side]
        if op_hand.get("tracked") is not True:
            raise AssertionError(f"sample op.hands.{side}.tracked must be true")
        if len(op_hand.get("wrist_body", [])) != 3 or len(op_hand.get("raw_wrist", [])) != 3:
            raise AssertionError(f"sample op.hands.{side} vector shape mismatch")
    status = sample.get("status", {})
    for key in ("engaged", "tracked"):
        if set(status.get(key, {}).keys()) != set(SIDES):
            raise AssertionError(f"sample status.{key} must include both sides")


def _assert_json_finite(value, path: str = "sample") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AssertionError(f"{path} must not contain NaN or Infinity")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_json_finite(child, f"{path}.{key}")
        return
    if isinstance(value, list):
        for i, child in enumerate(value):
            _assert_json_finite(child, f"{path}[{i}]")


def _assert_unity_manifest(manifest: dict) -> None:
    if not isinstance(manifest.get("dependencies"), dict):
        raise AssertionError("Unity manifest must contain a dependencies object")
    if manifest.get("dependencies"):
        raise AssertionError("Unity scaffold must stay dependency-free; primitive renderer should not require packages")


def _assert_unity_runner(unity_runner: str) -> None:
    for snippet in (
        'PROJECT = REPO / "unity" / "TeleopRenderer"',
        "SUCCESS_MARKER = \"TeleopRenderer editor validation passed\"",
        '"-projectPath", str(PROJECT)',
        '"-executeMethod", "TeleopEditorValidation.Run"',
        "log_contains(log_file, SUCCESS_MARKER)",
        "return 3",
        "DEFAULT_TIMEOUT_SECONDS = 180.0",
        "ap.add_argument(\"--timeout-seconds\"",
        "subprocess.run(cmd, cwd=REPO, timeout=args.timeout_seconds)",
        "except subprocess.TimeoutExpired",
        "return 4",
    ):
        if snippet not in unity_runner:
            raise AssertionError(f"run_unity_validation.py missing {snippet}")


def _assert_unity_gitignore(gitignore: str) -> None:
    for entry in (
        "unity/TeleopRenderer/Library/",
        "unity/TeleopRenderer/Temp/",
        "unity/TeleopRenderer/Obj/",
        "unity/TeleopRenderer/Logs/",
        "unity/TeleopRenderer/UserSettings/",
        "unity/TeleopRenderer/Build/",
        "unity/TeleopRenderer/Builds/",
    ):
        if entry not in gitignore:
            raise AssertionError(f".gitignore must ignore Unity generated path {entry}")


def _unity_asset_paths() -> list[Path]:
    assets = UNITY / "Assets"
    paths = [assets]
    if not assets.exists():
        return paths
    for path in assets.rglob("*"):
        if path.name.endswith(".meta"):
            continue
        if path.is_dir() or path.is_file():
            paths.append(path)
    return sorted(set(paths))


def _asset_meta_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.meta")


def _assert_unity_meta_files() -> None:
    seen_guids: dict[str, Path] = {}
    for asset in _unity_asset_paths():
        meta = _asset_meta_path(asset)
        if not meta.exists():
            raise AssertionError(f"missing Unity .meta file for {asset.relative_to(UNITY)}")
        text = _read(meta)
        if "fileFormatVersion: 2" not in text:
            raise AssertionError(f"{meta.relative_to(UNITY)} must use fileFormatVersion: 2")
        m = re.search(r"^guid:\s*([0-9a-f]{32})\s*$", text, re.MULTILINE)
        if not m:
            raise AssertionError(f"{meta.relative_to(UNITY)} must contain a 32-character hex guid")
        guid = m.group(1)
        if guid in seen_guids:
            other = seen_guids[guid].relative_to(UNITY)
            raise AssertionError(f"{meta.relative_to(UNITY)} duplicates Unity guid from {other}")
        seen_guids[guid] = meta
        if asset.is_dir():
            if "folderAsset: yes" not in text or "DefaultImporter:" not in text:
                raise AssertionError(f"{meta.relative_to(UNITY)} must be a Unity folder meta file")
        elif asset.suffix == ".cs":
            if "MonoImporter:" not in text:
                raise AssertionError(f"{meta.relative_to(UNITY)} must use MonoImporter")
        elif "DefaultImporter:" not in text:
            raise AssertionError(f"{meta.relative_to(UNITY)} must use DefaultImporter")


def _assert_unity_client_schema(client: str) -> None:
    m = re.search(r"\bExpectedSchemaVersion\s*=\s*(\d+)\s*;", client)
    if not m:
        raise AssertionError("TeleopRenderClient must declare ExpectedSchemaVersion")
    got = int(m.group(1))
    if got != topics.SCHEMA_VERSION:
        raise AssertionError(
            f"TeleopRenderClient ExpectedSchemaVersion {got} must match Python schema {topics.SCHEMA_VERSION}"
        )
    if "SupportedSchema" not in client or "state.v == ExpectedSchemaVersion" not in client:
        raise AssertionError("TeleopRenderClient must fail closed on render-state schema version mismatch")


def _unity_json_host_port(rig: dict) -> tuple[str, int]:
    endpoint = rig.get("vr", {}).get("unity_json_endpoint")
    if not isinstance(endpoint, str) or not endpoint.startswith("tcp://"):
        raise AssertionError("rig vr.unity_json_endpoint must be a tcp://host:port string")
    host, port_s = endpoint.removeprefix("tcp://").rsplit(":", 1)
    return host, int(port_s)


def _assert_unity_client_endpoint(client: str, rig: dict) -> None:
    host, port = _unity_json_host_port(rig)
    if f'public string host = "{host}";' not in client or f"public int port = {port};" not in client:
        raise AssertionError("TeleopRenderClient defaults must match rig vr.unity_json_endpoint")


def _assert_renderer_initializes_before_apply(src: str, class_name: str) -> None:
    if "private void EnsureInitialized()" not in src or "EnsureInitialized();" not in src:
        raise AssertionError(f"{class_name} must initialize safely before Apply")
    body = _class_body(src, class_name)
    m = re.search(r"\bpublic\s+void\s+Apply\s*\([^\)]*\)\s*\{", body)
    if not m:
        raise AssertionError(f"{class_name} missing public Apply")
    apply_body = body[m.end():]
    if "EnsureInitialized();" not in apply_body[:160]:
        raise AssertionError(f"{class_name}.Apply must call EnsureInitialized before touching primitives")


def _assert_unity_materials(materials: str) -> None:
    for snippet in (
        "public static class TeleopUnityMaterials",
        "\"Standard\"",
        "\"Universal Render Pipeline/Lit\"",
        "\"Unlit/Color\"",
        "\"Sprites/Default\"",
        "\"Hidden/Internal-Colored\"",
        "new Material(shader)",
        "mat.color = color",
    ):
        if snippet not in materials:
            raise AssertionError(f"TeleopUnityMaterials missing {snippet}")


def _assert_bootstrap_runtime_config(bootstrap: str) -> None:
    for snippet in (
        "public static void ConfigureRuntime()",
        "QualitySettings.vSyncCount = 0",
        "Application.targetFrameRate = 72",
        "Screen.sleepTimeout = SleepTimeout.NeverSleep",
    ):
        if snippet not in bootstrap:
            raise AssertionError(f"TeleopSceneBootstrap missing runtime config: {snippet}")
    body = _class_body(bootstrap, "TeleopSceneBootstrap")
    for method in ("CreateIfNeeded", "CreateRendererRoot"):
        m = re.search(rf"\b(?:private|public)\s+static\s+(?:void|GameObject)\s+{method}\s*\([^\)]*\)\s*\{{", body)
        if not m:
            raise AssertionError(f"TeleopSceneBootstrap missing {method}")
        method_body = body[m.end():]
        if not method_body.lstrip().startswith("ConfigureRuntime();"):
            raise AssertionError(f"TeleopSceneBootstrap.{method} must configure runtime before creating/rendering")
    create_body = _class_body(bootstrap, "TeleopSceneBootstrap")
    m = re.search(r"if\s*\(\s*FindObjectOfType<TeleopRenderClient>\(\)\s*!=\s*null\s*\)\s*\{(?P<body>.*?)\}", create_body, re.DOTALL)
    if not m or "EnsureSceneSupportObjects();" not in m.group("body"):
        raise AssertionError("TeleopSceneBootstrap must ensure scene support objects when a TeleopRenderClient already exists")


def _assert_status_hud(hud: str) -> None:
    for snippet in (
        "public sealed class TeleopStatusHud : MonoBehaviour",
        "public void Apply(RenderState state, string connectionStatus, string endpoint, float now)",
        "public void Clear(string connectionStatus, string endpoint, float now)",
        "public bool DebugHasState()",
        "public string DebugLine(int index)",
        "private void OnGUI()",
        "GUI.Box(panel, GUIContent.none)",
        "GUI.Label",
        "CalibrationLine(state.status.calib)",
        "OperatorLine(state.op)",
        "CommandErrorLine(state.arms)",
        "cmd_err L=",
        "Mathf.Sqrt(dx * dx + dy * dy + dz * dz) * 100.0f",
        "\" engaged=\" + Flags(state.status.engaged)",
        "\" tracked=\" + Flags(state.status.tracked)",
        "operator=head:",
    ):
        if snippet not in hud:
            raise AssertionError(f"TeleopStatusHud missing {snippet}")
    if "UnityEngine.UI" in hud:
        raise AssertionError("TeleopStatusHud must stay dependency-free and use built-in IMGUI")


def main() -> int:
    for path in REQUIRED_FILES:
        if not path.exists():
            raise AssertionError(f"missing Unity scaffold file: {path.relative_to(REPO)}")

    manifest = json.loads(_read(UNITY / "Packages" / "manifest.json"))
    _assert_unity_manifest(manifest)

    project_version = _read(UNITY / "ProjectSettings" / "ProjectVersion.txt")
    if not re.search(r"^m_EditorVersion:\s*\d+\.\d+\.\d+f\d+", project_version, re.MULTILINE):
        raise AssertionError("Unity ProjectVersion.txt must pin a Unity editor version")
    _assert_sample_payload(json.loads(_read(SAMPLE)))
    _assert_unity_gitignore(_read(GITIGNORE))
    _assert_unity_meta_files()
    if _read(SAMPLE) != normalized_text(make_fixture()):
        raise AssertionError("Unity render-state fixture is stale; run scripts/update_unity_fixture.py --write")

    client = _read(SCRIPTS / "TeleopRenderClient.cs")
    arm = _read(SCRIPTS / "YamArmRenderer.cs")
    hand = _read(SCRIPTS / "OrcaHandRenderer.cs")
    op = _read(SCRIPTS / "OperatorVectorRenderer.cs")
    frame = _read(SCRIPTS / "TeleopUnityFrame.cs")
    materials = _read(SCRIPTS / "TeleopUnityMaterials.cs")
    hud = _read(SCRIPTS / "TeleopStatusHud.cs")
    editor_validation = _read(EDITOR / "TeleopEditorValidation.cs")
    scene_asset = _read(EDITOR / "TeleopSceneAsset.cs")
    unity_runner = _read(UNITY_RUNNER)
    cs_files = sorted(SCRIPTS.glob("*.cs")) + sorted(EDITOR.glob("*.cs"))
    sources = {p: _read(p) for p in cs_files}
    all_cs = "\n".join(sources.values())

    for path, src in sources.items():
        _assert_balanced(path, src)
        if "MonoBehaviour" in src:
            expected = path.stem
            if not re.search(rf"\bpublic\s+(?:sealed\s+)?class\s+{re.escape(expected)}\s*:\s*MonoBehaviour\b", src):
                raise AssertionError(f"{path.name} must define public MonoBehaviour class {expected}")

    if "Dictionary<" in all_cs:
        raise AssertionError("Unity JsonUtility DTOs must not use Dictionary<>")
    if "JsonUtility.FromJson<RenderState>" not in client:
        raise AssertionError("TeleopRenderClient must deserialize the Python render state")
    _assert_unity_client_endpoint(client, load_rig())
    if "Application.isPlaying" not in client:
        raise AssertionError("TeleopRenderClient must avoid starting sockets in Editor validation mode")
    if "public void ApplyState(RenderState state)" not in client or "Flag(engaged, YamSide.Left)" not in client:
        raise AssertionError("TeleopRenderClient must expose a null-tolerant state apply path for validation")
    if "public TeleopStatusHud statusHud" not in client or "UpdateStatusHud(now)" not in client \
            or "statusHud.Apply(latestState, status" not in client or "statusHud.Clear(status" not in client:
        raise AssertionError("TeleopRenderClient must update the Unity status HUD on render-state changes")
    _assert_unity_client_schema(client)
    if "ValidStateShape(RenderState state)" not in client or "&& state.arms.left != null" not in client \
            or "&& state.status.tracked != null" not in client:
        raise AssertionError("TeleopRenderClient must reject version-correct but incomplete render states")
    for snippet in (
        "ExpectedArmJointCount = 6",
        "ExpectedArmLinkFloatCount = 24",
        "ExpectedHandJointCount = 17",
        "ExpectedVec3FloatCount = 3",
        "ExpectedQuatFloatCount = 4",
        "FiniteArray(float[] values, int expectedLength)",
        "float.IsNaN(values[i])",
        "float.IsInfinity(values[i])",
        "ValidArmStateShape(state.arms.left)",
        "ValidArmStateShape(state.arms.right)",
        "ValidHandRenderShape(state.hand_render.left)",
        "ValidHandRenderShape(state.hand_render.right)",
        "ValidOperatorStateShape(state.op)",
        "FiniteArray(arm.q, ExpectedArmJointCount)",
        "FiniteArray(arm.link_pos, ExpectedArmLinkFloatCount)",
        "FiniteArray(arm.ee_pos, ExpectedVec3FloatCount)",
        "FiniteArray(arm.ee_quat, ExpectedQuatFloatCount)",
        "arm.cmd_pos == null || FiniteArray(arm.cmd_pos, ExpectedVec3FloatCount)",
        "FiniteArray(arm.margins, ExpectedArmJointCount)",
        "hand.names.Length == ExpectedHandJointCount",
        "FiniteArray(hand.q, ExpectedHandJointCount)",
        "Vec3(op.torso_from_head)",
        "ValidOperatorHandShape(op.hands.left)",
        "ValidOperatorHandShape(op.hands.right)",
        "hand.tracked ? Vec3(hand.wrist_body) : hand.wrist_body == null",
        "NullableVec3(hand.raw_wrist)",
        "ValidStatusShape(state.status)",
        "FiniteValue(status.hz)",
        "status.calib == null || ValidCalibrationShape(status.calib)",
        "FiniteValue(calib.progress)",
        "FiniteValue(calib.remaining)",
        "FiniteValue(float value)",
    ):
        if snippet not in client:
            raise AssertionError(f"TeleopRenderClient top-level shape gate missing {snippet}")
    if "stateTimeoutSeconds" not in client or "HideIfStale" not in client \
            or "HideRenderers()" not in client or 'status = "stale"' not in client:
        raise AssertionError("TeleopRenderClient must hide renderers when the Unity render stream goes stale")
    if "public void ApplyJsonAt(string json, float now)" not in client:
        raise AssertionError("TeleopRenderClient must expose JSON apply path for editor validation")
    if "RenderState parsed = JsonUtility.FromJson<RenderState>(json)" not in client \
            or "latestState = state" not in client or "latestState = null" not in client \
            or "DebugHasLatestState()" not in client:
        raise AssertionError("TeleopRenderClient must keep HUD latest-state bookkeeping in the apply/hide path")
    if "catch (Exception e)" not in client or "HideRenderers();" not in client \
            or 'status = "json error: " + e.Message' not in client:
        raise AssertionError("TeleopRenderClient must hide renderers on malformed JSON")

    _assert_fields(client, "RenderState", {"v", "stamp", "arms", "hand_render", "op", "status"})
    _assert_fields(client, "RenderArms", {"left", "right"})
    _assert_fields(client, "RenderArmState", {"q", "link_pos", "ee_pos", "ee_quat", "cmd_pos", "cmd_quat", "margins"})
    _assert_fields(client, "RenderHands", {"left", "right"})
    _assert_fields(client, "RenderHandState", {"names", "q"})
    _assert_fields(client, "RenderOperatorState", {"torso_from_head", "head_pos", "torso_pos", "hands"})
    _assert_fields(client, "RenderOperatorHands", {"left", "right"})
    _assert_fields(client, "RenderOperatorHand", {"tracked", "wrist_body", "raw_wrist"})
    _assert_fields(client, "RenderStatus", {"engaged", "tracked", "calib", "hz"})
    _assert_fields(client, "SideFlags", {"left", "right"})
    _assert_fields(client, "CalibrationStatus", {"active", "phase", "progress", "remaining", "left", "right", "msg"})

    if "DecodeLinkPoints(state.link_pos)" not in arm:
        raise AssertionError("YamArmRenderer must draw from Python-published link_pos")
    _assert_renderer_initializes_before_apply(arm, "YamArmRenderer")
    if "TeleopUnityMaterials.Make" not in arm:
        raise AssertionError("YamArmRenderer must use shared material creation")
    if "ExpectedLinkFloatCount = 24" not in arm or "ValidArmState" not in arm:
        raise AssertionError("YamArmRenderer must reject malformed fixed-shape arm payloads")
    if "FiniteArray(state.link_pos, ExpectedLinkFloatCount)" not in arm \
            or "FiniteArray(state.ee_pos, 3)" not in arm \
            or "state.cmd_pos == null || FiniteArray(state.cmd_pos, 3)" not in arm \
            or "float.IsNaN(values[i])" not in arm \
            or "float.IsInfinity(values[i])" not in arm:
        raise AssertionError("YamArmRenderer must reject non-finite arm payloads")
    if "DebugJointPosition" not in arm or "DebugEePosition" not in arm or "DebugCmdPosition" not in arm \
            or "DebugCmdLineStartPosition" not in arm or "DebugCmdLineEndPosition" not in arm \
            or "DebugJointActive" not in arm or "DebugCmdActive" not in arm or "DebugCmdLineActive" not in arm:
        raise AssertionError("YamArmRenderer must expose applied positions for editor validation")
    if "commanded EE target" not in arm or "TeleopUnityFrame.RobotWorldToUnity(new Vector3(state.cmd_pos[0]" not in arm:
        raise AssertionError("YamArmRenderer must draw the Python-published commanded EE target")
    if "command error" not in arm or "cmdErrorLine.SetPosition(0, eeMarker.transform.position)" not in arm \
            or "cmdErrorLine.SetPosition(1, cmdMarker.transform.position)" not in arm:
        raise AssertionError("YamArmRenderer must draw achieved-to-command error line")
    if "JointPlacement" in arm or "JointAxis" in arm:
        raise AssertionError("YamArmRenderer must not duplicate arm FK constants")
    if "TeleopUnityFrame.RobotWorldToUnity" not in arm:
        raise AssertionError("YamArmRenderer must use the shared robot-world to Unity conversion")
    if "RenderHandState" not in hand or "Value(hand," not in hand:
        raise AssertionError("OrcaHandRenderer must draw from fixed hand_render names/q")
    _assert_renderer_initializes_before_apply(hand, "OrcaHandRenderer")
    if "TeleopUnityMaterials.Make" not in hand:
        raise AssertionError("OrcaHandRenderer must use shared material creation")
    if "ExpectedJointCount = 17" not in hand or "ValidHandState" not in hand:
        raise AssertionError("OrcaHandRenderer must reject malformed fixed-shape hand payloads")
    if "FiniteArray(hand.q, ExpectedJointCount)" not in hand \
            or "FiniteArray(arm.ee_pos, 3)" not in hand \
            or "FiniteArray(arm.ee_quat, 4)" not in hand \
            or "float.IsNaN(values[i])" not in hand \
            or "float.IsInfinity(values[i])" not in hand:
        raise AssertionError("OrcaHandRenderer must reject non-finite hand/EE payloads")
    if "RenderArmState" not in hand or "arm.ee_pos" not in hand:
        raise AssertionError("OrcaHandRenderer must anchor hands to the achieved arm EE pose")
    if "DebugPalmPosition" not in hand or "DebugPalmRotation" not in hand or "DebugPalmActive" not in hand:
        raise AssertionError("OrcaHandRenderer must expose applied palm pose for editor validation")
    if "TeleopUnityFrame.RobotWorldToUnity" not in hand or "TeleopUnityFrame.RobotQuatToUnity(arm.ee_quat)" not in hand:
        raise AssertionError("OrcaHandRenderer must use the shared achieved EE pose conversion")
    if "TeleopUnityFrame.BodyVectorToUnity(hand.wrist_body)" not in op:
        raise AssertionError("OperatorVectorRenderer must draw body-frame torso-to-wrist vectors through the shared boundary helper")
    if "ExpectedBodyVectorFloatCount = 3" not in op or "ValidHand" not in op \
            or "FiniteArray(hand.wrist_body, ExpectedBodyVectorFloatCount)" not in op:
        raise AssertionError("OperatorVectorRenderer must reject malformed fixed-shape wrist_body payloads")
    if "float.IsNaN(values[i])" not in op or "float.IsInfinity(values[i])" not in op:
        raise AssertionError("OperatorVectorRenderer must reject non-finite wrist_body payloads")
    _assert_renderer_initializes_before_apply(op, "OperatorVectorRenderer")
    if "TeleopUnityMaterials.Make" not in op:
        raise AssertionError("OperatorVectorRenderer must use shared material creation")
    if "obj.transform.SetParent(transform, false)" not in op:
        raise AssertionError("OperatorVectorRenderer line objects must be parented under the overlay")
    if "DebugLeftWristPosition" not in op or "DebugRightWristPosition" not in op \
            or "DebugLeftLineEndPosition" not in op or "DebugRightLineEndPosition" not in op \
            or "DebugLeftWristActive" not in op or "DebugRightWristActive" not in op \
            or "DebugLeftLineActive" not in op or "DebugRightLineActive" not in op:
        raise AssertionError("OperatorVectorRenderer must expose applied wrist vector for editor validation")
    for src_name, src in (("YamArmRenderer", arm), ("OrcaHandRenderer", hand), ("OperatorVectorRenderer", op)):
        if re.search(r"\bprivate\s+static\s+Vector3\s+RobotWorldToUnity\s*\(", src) \
                or re.search(r"\bprivate\s+static\s+Quaternion\s+RobotQuatToUnity\s*\(", src) \
                or re.search(r"\bprivate\s+static\s+Vector3\s+BodyToUnity\s*\(", src):
            raise AssertionError(f"{src_name} must not define a private coordinate conversion")
        if "Shader.Find" in src or "new Material(" in src:
            raise AssertionError(f"{src_name} must use TeleopUnityMaterials instead of local shader/material setup")
    for snippet in (
        "public static class TeleopUnityFrame",
        "new Vector3(p.y, p.z, -p.x)",
        "new Quaternion(q[1], q[2], q[3], q[0])",
        "Vector3.Cross(right.normalized, up.normalized)",
        "Quaternion.LookRotation(forward.normalized, up.normalized)",
        "new Vector3(body[0], body[1], body[2])",
    ):
        if snippet not in frame:
            raise AssertionError(f"TeleopUnityFrame missing {snippet}")
    _assert_unity_materials(materials)
    _assert_status_hud(hud)
    for snippet in (
        "TeleopEditorValidation",
        "JsonUtility.FromJson<RenderState>",
        "render_state_sample.json",
        "File.ReadAllText",
        "ValidateMaterialFactory",
        "TeleopUnityMaterials.Make",
        "material factory returned material without shader",
        "material factory did not preserve color",
        "ColorClose",
        "ValidateBootstrapRuntimeConfig",
        "TeleopSceneBootstrap.ConfigureRuntime()",
        "QualitySettings.vSyncCount == 0",
        "Application.targetFrameRate == 72",
        "Screen.sleepTimeout == SleepTimeout.NeverSleep",
        "bootstrap runtime did not disable vSync",
        "bootstrap runtime did not target 72 FPS",
        "bootstrap runtime did not prevent sleep",
        "left link_pos shape",
        "right link_pos shape",
        "left hand_render shape",
        "right hand_render shape",
        "left operator wrist_body shape",
        "right operator wrist_body shape",
        "ValidateCoordinateFrame",
        "TeleopUnityFrame.RobotWorldToUnity",
        "TeleopUnityFrame.RobotQuatToUnity",
        "TeleopUnityFrame.BodyVectorToUnity",
        "ValidateBootstrapWiring",
        "bootstrap duplicated existing camera",
        "bootstrap duplicated existing light",
        "bootstrap duplicated MainCamera tag",
        "bootstrap duplicated existing floor",
        "ValidateSceneAssetAndBuildSettings",
        "TeleopSceneAsset.EnsureRendererScene()",
        "BuildSettingsContainRendererScene()",
        "CreateRendererRoot",
        "leftArm.Apply",
        "rightArm.Apply",
        "leftHand.Apply",
        "rightHand.Apply",
        "overlay.Apply",
        "DebugJointPosition(0)",
        "DebugEePosition()",
        "DebugCmdPosition()",
        "DebugCmdLineStartPosition()",
        "DebugCmdLineEndPosition()",
        "left arm did not apply cmd_pos",
        "right arm did not apply cmd_pos",
        "left arm did not draw command error line",
        "right arm did not draw command error line",
        "ApplyState(sample)",
        "v = sample.v + 1",
        "schema version mismatch did not hide renderers",
        "ApplyState(new RenderState {v = sample.v})",
        "client partial state did not hide renderers or clear latest state",
        "client malformed arm state did not hide renderers or clear latest state",
        "client malformed hand_render state did not hide renderers or clear latest state",
        "client malformed operator state did not hide renderers or clear latest state",
        "client non-finite arm state did not hide renderers or clear latest state",
        "client non-finite operator state did not hide renderers or clear latest state",
        "client non-finite torso_from_head did not hide renderers or clear latest state",
        "client non-finite status hz did not hide renderers or clear latest state",
        "client non-finite calibration status did not hide renderers or clear latest state",
        "float.NaN",
        "float.PositiveInfinity",
        "ApplyJsonAt(\"{not valid json\", 11.0f)",
        "malformed json did not hide renderers or clear latest state",
        "DebugJointActive(0)",
        "malformed arm link_pos did not hide left arm",
        "malformed arm link_pos did not hide right arm",
        "non-finite arm ee_pos did not hide left arm",
        "non-finite arm ee_pos did not hide right arm",
        "null cmd_pos did not hide left command marker/line only",
        "non-finite arm cmd_pos did not hide left arm",
        "DebugPalmPosition()",
        "DebugPalmActive()",
        "malformed hand_render did not hide left hand",
        "malformed hand_render did not hide right hand",
        "non-finite hand_render did not hide left hand",
        "non-finite hand_render did not hide right hand",
        "DebugLeftWristPosition()",
        "DebugRightWristPosition()",
        "DebugLeftLineEndPosition()",
        "DebugRightLineEndPosition()",
        "DebugLeftWristActive()",
        "DebugRightWristActive()",
        "DebugLeftLineActive()",
        "DebugRightLineActive()",
        "operator overlay left wrist_body null/untracked did not hide",
        "operator overlay right wrist_body null/tracked did not hide",
        "operator overlay malformed wrist_body did not hide left",
        "operator overlay malformed left wrist_body hid right",
        "operator overlay malformed right wrist_body hid left",
        "operator overlay malformed wrist_body did not hide right",
        "operator overlay non-finite wrist_body did not hide left",
        "client.rightHand = rightHand",
        "client.statusHud = hud",
        "ApplyStateAt(sample, 10.0f)",
        "client.DebugHasLatestState()",
        "hud.DebugHasState()",
        "status HUD did not show accepted render state",
        "status HUD did not show command error",
        "HideIfStale(10.0f + client.stateTimeoutSeconds + 0.1f)",
        "stale render state did not hide renderers or clear latest state",
        "status HUD did not show stale state",
        "schema version mismatch did not hide renderers or clear latest state",
        "status HUD did not show schema mismatch",
        "status HUD did not show malformed JSON",
        "client.host == \"127.0.0.1\"",
        "client.port == 8102",
    ):
        if snippet not in editor_validation:
            raise AssertionError(f"TeleopEditorValidation missing {snippet}")

    for snippet in (
        "public static class TeleopSceneAsset",
        "ScenePath = \"Assets/Scenes/TeleopRenderer.unity\"",
        "EnsureRendererScene",
        "EditorSceneManager.NewScene",
        "TeleopSceneBootstrap.CreateRendererRoot()",
        "EditorSceneManager.SaveScene(scene, ScenePath)",
        "EditorBuildSettings.scenes",
        "BuildSettingsContainRendererScene",
    ):
        if snippet not in scene_asset:
            raise AssertionError(f"TeleopSceneAsset missing {snippet}")

    _assert_unity_runner(unity_runner)

    bootstrap = _read(SCRIPTS / "TeleopSceneBootstrap.cs")
    _assert_bootstrap_runtime_config(bootstrap)
    for snippet in (
        "CreateRendererRoot()",
        "AddComponent<TeleopRenderClient>()",
        "AddComponent<YamArmRenderer>()",
        "AddComponent<OrcaHandRenderer>()",
        "AddComponent<OperatorVectorRenderer>()",
        "AddComponent<TeleopStatusHud>()",
        "client.leftHand = leftHand",
        "client.rightHand = rightHand",
        "client.statusHud = hud",
        "public static void EnsureSceneSupportObjects()",
        "Camera.main != null ? Camera.main : GameObject.FindObjectOfType<Camera>()",
        "GameObject.FindObjectOfType<Light>()",
        "GameObject.Find(\"Floor\")",
        "TeleopUnityMaterials.Make",
    ):
        if snippet not in bootstrap:
            raise AssertionError(f"TeleopSceneBootstrap missing {snippet}")
    if "Shader.Find" in bootstrap or "new Material(" in bootstrap:
        raise AssertionError("TeleopSceneBootstrap must use TeleopUnityMaterials for scene materials")

    print("Unity render contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
