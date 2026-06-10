from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from bimanual_teleop.bus import topics


def _load_contract():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    script = scripts / "check_unity_contract.py"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("check_unity_contract", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_unity_contract_accepts_dependency_free_manifest():
    contract = _load_contract()
    contract._assert_unity_manifest({"dependencies": {}})


def test_unity_contract_rejects_package_dependencies():
    contract = _load_contract()
    with pytest.raises(AssertionError, match="dependency-free"):
        contract._assert_unity_manifest({"dependencies": {"com.unity.inputsystem": "1.7.0"}})


def test_unity_contract_rejects_missing_project_path_check():
    contract = _load_contract()
    with pytest.raises(AssertionError, match="PROJECT"):
        contract._assert_unity_runner('SUCCESS_MARKER = "TeleopRenderer editor validation passed"\nreturn 3\n')


def test_unity_contract_accepts_current_runner():
    contract = _load_contract()
    runner = (Path(__file__).resolve().parents[1] / "scripts" / "run_unity_validation.py").read_text(encoding="utf-8")
    contract._assert_unity_runner(runner)


def test_unity_contract_accepts_unity_gitignore_entries():
    contract = _load_contract()
    contract._assert_unity_gitignore(
        "\n".join([
            "unity/TeleopRenderer/Library/",
            "unity/TeleopRenderer/Temp/",
            "unity/TeleopRenderer/Obj/",
            "unity/TeleopRenderer/Logs/",
            "unity/TeleopRenderer/UserSettings/",
            "unity/TeleopRenderer/Build/",
            "unity/TeleopRenderer/Builds/",
        ])
    )


def test_unity_contract_rejects_missing_unity_gitignore_entry():
    contract = _load_contract()
    with pytest.raises(AssertionError, match="Unity generated path"):
        contract._assert_unity_gitignore("unity/TeleopRenderer/Library/\n")


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _folder_meta(guid: str) -> str:
    return (
        "fileFormatVersion: 2\n"
        f"guid: {guid}\n"
        "folderAsset: yes\n"
        "DefaultImporter:\n"
        "  externalObjects: {}\n"
    )


def _mono_meta(guid: str) -> str:
    return (
        "fileFormatVersion: 2\n"
        f"guid: {guid}\n"
        "MonoImporter:\n"
        "  externalObjects: {}\n"
    )


def _default_meta(guid: str) -> str:
    return (
        "fileFormatVersion: 2\n"
        f"guid: {guid}\n"
        "DefaultImporter:\n"
        "  externalObjects: {}\n"
    )


def test_unity_contract_accepts_complete_meta_files(tmp_path, monkeypatch):
    contract = _load_contract()
    monkeypatch.setattr(contract, "UNITY", tmp_path)
    _write(tmp_path / "Assets.meta", _folder_meta("11111111111111111111111111111111"))
    _write(tmp_path / "Assets" / "Scripts.meta", _folder_meta("22222222222222222222222222222222"))
    _write(tmp_path / "Assets" / "Scripts" / "Foo.cs", "public sealed class Foo {}\n")
    _write(tmp_path / "Assets" / "Scripts" / "Foo.cs.meta", _mono_meta("33333333333333333333333333333333"))
    _write(tmp_path / "Assets" / "Editor.meta", _folder_meta("44444444444444444444444444444444"))
    _write(tmp_path / "Assets" / "Editor" / "sample.json", "{}\n")
    _write(tmp_path / "Assets" / "Editor" / "sample.json.meta", _default_meta("55555555555555555555555555555555"))

    contract._assert_unity_meta_files()


def test_unity_contract_rejects_missing_meta_file(tmp_path, monkeypatch):
    contract = _load_contract()
    monkeypatch.setattr(contract, "UNITY", tmp_path)
    _write(tmp_path / "Assets.meta", _folder_meta("11111111111111111111111111111111"))
    _write(tmp_path / "Assets" / "Scripts" / "Foo.cs", "public sealed class Foo {}\n")

    with pytest.raises(AssertionError, match="missing Unity \\.meta file"):
        contract._assert_unity_meta_files()


def test_unity_contract_rejects_duplicate_or_malformed_meta_guid(tmp_path, monkeypatch):
    contract = _load_contract()
    monkeypatch.setattr(contract, "UNITY", tmp_path)
    _write(tmp_path / "Assets.meta", _folder_meta("11111111111111111111111111111111"))
    (tmp_path / "Assets" / "Scripts").mkdir(parents=True)
    _write(tmp_path / "Assets" / "Scripts.meta", _folder_meta("11111111111111111111111111111111"))

    with pytest.raises(AssertionError, match="duplicates Unity guid"):
        contract._assert_unity_meta_files()

    _write(tmp_path / "Assets" / "Scripts.meta", _folder_meta("not-a-guid"))
    with pytest.raises(AssertionError, match="32-character hex guid"):
        contract._assert_unity_meta_files()


def test_unity_contract_accepts_matching_client_schema_version():
    contract = _load_contract()
    contract._assert_unity_client_schema(
        f"private const int ExpectedSchemaVersion = {topics.SCHEMA_VERSION};\n"
        "private static bool SupportedSchema(RenderState state) { return state.v == ExpectedSchemaVersion; }\n"
    )


def test_unity_contract_rejects_stale_client_schema_version():
    contract = _load_contract()
    with pytest.raises(AssertionError, match="must match Python schema"):
        contract._assert_unity_client_schema(
            f"private const int ExpectedSchemaVersion = {topics.SCHEMA_VERSION + 1};\n"
            "private static bool SupportedSchema(RenderState state) { return state.v == ExpectedSchemaVersion; }\n"
        )


def test_unity_contract_rejects_missing_client_schema_gate():
    contract = _load_contract()
    with pytest.raises(AssertionError, match="ExpectedSchemaVersion"):
        contract._assert_unity_client_schema("private static bool SupportedSchema(RenderState state) { return true; }\n")


def test_unity_contract_accepts_client_endpoint_from_rig():
    contract = _load_contract()
    rig = {"vr": {"unity_json_endpoint": "tcp://192.168.1.10:9000"}}
    contract._assert_unity_client_endpoint('public string host = "192.168.1.10";\npublic int port = 9000;\n', rig)


def test_unity_contract_rejects_client_endpoint_mismatch():
    contract = _load_contract()
    rig = {"vr": {"unity_json_endpoint": "tcp://127.0.0.1:8102"}}
    with pytest.raises(AssertionError, match="unity_json_endpoint"):
        contract._assert_unity_client_endpoint('public string host = "127.0.0.1";\npublic int port = 9000;\n', rig)


def test_unity_contract_rejects_invalid_rig_unity_endpoint():
    contract = _load_contract()
    with pytest.raises(AssertionError, match="tcp://host:port"):
        contract._unity_json_host_port({"vr": {"unity_json_endpoint": "udp://127.0.0.1:8102"}})


def test_unity_contract_accepts_material_factory_helper():
    contract = _load_contract()
    materials = (
        "public static class TeleopUnityMaterials {\n"
        '  private static readonly string[] ShaderNames = {"Standard", "Universal Render Pipeline/Lit", "Unlit/Color", "Sprites/Default"};\n'
        '  public static Material Make(Color color) { var shader = Shader.Find("Hidden/Internal-Colored"); var mat = new Material(shader); mat.color = color; return mat; }\n'
        "}\n"
    )
    contract._assert_unity_materials(materials)


def test_unity_contract_rejects_material_factory_without_fallback_shader():
    contract = _load_contract()
    with pytest.raises(AssertionError, match="Hidden/Internal-Colored"):
        contract._assert_unity_materials(
            "public static class TeleopUnityMaterials {\n"
            '  private static readonly string[] ShaderNames = {"Standard", "Universal Render Pipeline/Lit", "Unlit/Color", "Sprites/Default"};\n'
            "  public static Material Make(Color color) { var mat = new Material(shader); mat.color = color; return mat; }\n"
            "}\n"
        )


def test_unity_contract_accepts_bootstrap_runtime_config():
    contract = _load_contract()
    contract._assert_bootstrap_runtime_config(
        "using UnityEngine;\n"
        "public sealed class TeleopSceneBootstrap : MonoBehaviour {\n"
        "  private static void CreateIfNeeded() { ConfigureRuntime(); if (FindObjectOfType<TeleopRenderClient>() != null) { EnsureSceneSupportObjects(); return; } CreateRendererRoot(); }\n"
        "  public static GameObject CreateRendererRoot() { ConfigureRuntime(); return new GameObject(); }\n"
        "  public static void EnsureSceneSupportObjects() {}\n"
        "  public static void ConfigureRuntime() {\n"
        "    QualitySettings.vSyncCount = 0;\n"
        "    Application.targetFrameRate = 72;\n"
        "    Screen.sleepTimeout = SleepTimeout.NeverSleep;\n"
        "  }\n"
        "}\n"
    )


def test_unity_contract_rejects_bootstrap_existing_client_without_support_objects():
    contract = _load_contract()
    with pytest.raises(AssertionError, match="already exists"):
        contract._assert_bootstrap_runtime_config(
            "using UnityEngine;\n"
            "public sealed class TeleopSceneBootstrap : MonoBehaviour {\n"
            "  private static void CreateIfNeeded() { ConfigureRuntime(); if (FindObjectOfType<TeleopRenderClient>() != null) { return; } CreateRendererRoot(); }\n"
            "  public static GameObject CreateRendererRoot() { ConfigureRuntime(); return new GameObject(); }\n"
            "  public static void EnsureSceneSupportObjects() {}\n"
            "  public static void ConfigureRuntime() {\n"
            "    QualitySettings.vSyncCount = 0;\n"
            "    Application.targetFrameRate = 72;\n"
            "    Screen.sleepTimeout = SleepTimeout.NeverSleep;\n"
            "  }\n"
            "}\n"
        )


def test_unity_contract_rejects_bootstrap_without_runtime_config():
    contract = _load_contract()
    with pytest.raises(AssertionError, match="runtime config"):
        contract._assert_bootstrap_runtime_config(
            "using UnityEngine;\n"
            "public sealed class TeleopSceneBootstrap : MonoBehaviour {\n"
            "  private static void CreateIfNeeded() { CreateRendererRoot(); }\n"
            "  public static GameObject CreateRendererRoot() { return new GameObject(); }\n"
            "}\n"
        )


def test_unity_contract_rejects_bootstrap_that_configures_too_late():
    contract = _load_contract()
    with pytest.raises(AssertionError, match="CreateIfNeeded"):
        contract._assert_bootstrap_runtime_config(
            "using UnityEngine;\n"
            "public sealed class TeleopSceneBootstrap : MonoBehaviour {\n"
            "  private static void CreateIfNeeded() { CreateRendererRoot(); ConfigureRuntime(); }\n"
            "  public static GameObject CreateRendererRoot() { ConfigureRuntime(); return new GameObject(); }\n"
            "  public static void ConfigureRuntime() {\n"
            "    QualitySettings.vSyncCount = 0;\n"
            "    Application.targetFrameRate = 72;\n"
            "    Screen.sleepTimeout = SleepTimeout.NeverSleep;\n"
            "  }\n"
            "}\n"
        )


def test_unity_contract_accepts_status_hud_helper():
    contract = _load_contract()
    contract._assert_status_hud(
        "using UnityEngine;\n"
        "public sealed class TeleopStatusHud : MonoBehaviour {\n"
        "  public void Apply(RenderState state, string connectionStatus, string endpoint, float now) {\n"
        "    line2 = \"hz=\" + state.status.hz.ToString(\"F1\") + \" engaged=\" + Flags(state.status.engaged) + \" tracked=\" + Flags(state.status.tracked);\n"
        "    line3 = CalibrationLine(state.status.calib);\n"
        "    line4 = OperatorLine(state.op);\n"
        "    line5 = CommandErrorLine(state.arms);\n"
        "  }\n"
        "  public void Clear(string connectionStatus, string endpoint, float now) {}\n"
        "  public bool DebugHasState() { return false; }\n"
        "  public string DebugLine(int index) { return \"operator=head: cmd_err L=\"; }\n"
        "  private void OnGUI() { GUI.Box(panel, GUIContent.none); GUI.Label(rect, line0); }\n"
        "  string CommandErrorLine(RenderArms arms) { return \"cmd_err L=\" + CommandErrorCm(arms.left); }\n"
        "  string CommandErrorCm(RenderArmState arm) { float dx = 0f; float dy = 0f; float dz = 0f; return (Mathf.Sqrt(dx * dx + dy * dy + dz * dz) * 100.0f).ToString(); }\n"
        "}\n"
    )


def test_unity_contract_rejects_status_hud_ui_dependency():
    contract = _load_contract()
    with pytest.raises(AssertionError, match="dependency-free"):
        contract._assert_status_hud(
            "using UnityEngine.UI;\n"
            "public sealed class TeleopStatusHud : MonoBehaviour {\n"
            "  public void Apply(RenderState state, string connectionStatus, string endpoint, float now) {}\n"
            "  public void Clear(string connectionStatus, string endpoint, float now) {}\n"
            "  public bool DebugHasState() { return false; }\n"
            "  public string DebugLine(int index) { return \"operator=head: cmd_err L=\"; }\n"
            "  private void OnGUI() { GUI.Box(panel, GUIContent.none); GUI.Label(rect, line0); }\n"
            "  void Render(RenderState state) { CalibrationLine(state.status.calib); OperatorLine(state.op); CommandErrorLine(state.arms); string x = \" engaged=\" + Flags(state.status.engaged) + \" tracked=\" + Flags(state.status.tracked); }\n"
            "  string CommandErrorLine(RenderArms arms) { return \"cmd_err L=\" + CommandErrorCm(arms.left); }\n"
            "  string CommandErrorCm(RenderArmState arm) { float dx = 0f; float dy = 0f; float dz = 0f; return (Mathf.Sqrt(dx * dx + dy * dy + dz * dz) * 100.0f).ToString(); }\n"
            "}\n"
        )


def test_unity_fixture_normalized_text_rejects_non_finite_json():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    script = scripts / "update_unity_fixture.py"
    spec = importlib.util.spec_from_file_location("update_unity_fixture", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with pytest.raises(ValueError):
        mod.normalized_text({"v": 1, "bad": math.nan})


def test_unity_contract_rejects_non_finite_sample_payload():
    contract = _load_contract()
    with pytest.raises(AssertionError, match="NaN or Infinity"):
        contract._assert_sample_payload({"v": topics.SCHEMA_VERSION, "arms": {"left": {"q": [math.nan]}}})
