#if UNITY_EDITOR
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class TeleopSceneAsset
{
    public const string ScenePath = "Assets/Scenes/TeleopRenderer.unity";

    [MenuItem("Teleop/Ensure Renderer Scene")]
    public static void EnsureRendererSceneFromMenu()
    {
        EnsureRendererScene();
    }

    public static void EnsureRendererScene()
    {
        string sceneDir = Path.Combine(Application.dataPath, "Scenes");
        Directory.CreateDirectory(sceneDir);

        Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        TeleopSceneBootstrap.CreateRendererRoot();
        if (!EditorSceneManager.SaveScene(scene, ScenePath))
        {
            throw new System.Exception("Failed to save " + ScenePath);
        }

        EnsureBuildSettingsScene();
        AssetDatabase.Refresh();
    }

    public static bool BuildSettingsContainRendererScene()
    {
        foreach (var scene in EditorBuildSettings.scenes)
        {
            if (scene.path == ScenePath && scene.enabled)
            {
                return true;
            }
        }
        return false;
    }

    private static void EnsureBuildSettingsScene()
    {
        var scenes = new List<EditorBuildSettingsScene>(EditorBuildSettings.scenes);
        bool found = false;
        for (int i = 0; i < scenes.Count; i++)
        {
            if (scenes[i].path == ScenePath)
            {
                scenes[i] = new EditorBuildSettingsScene(ScenePath, true);
                found = true;
            }
        }

        if (!found)
        {
            scenes.Add(new EditorBuildSettingsScene(ScenePath, true));
        }

        EditorBuildSettings.scenes = scenes.ToArray();
    }
}
#endif
