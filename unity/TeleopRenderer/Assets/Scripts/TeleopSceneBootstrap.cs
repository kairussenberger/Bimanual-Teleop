using UnityEngine;

public sealed class TeleopSceneBootstrap : MonoBehaviour
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void CreateIfNeeded()
    {
        ConfigureRuntime();
        if (FindObjectOfType<TeleopRenderClient>() != null)
        {
            EnsureSceneSupportObjects();
            return;
        }

        CreateRendererRoot();
    }

    public static GameObject CreateRendererRoot()
    {
        ConfigureRuntime();
        var root = new GameObject("TeleopRenderer");
        var client = root.AddComponent<TeleopRenderClient>();

        var left = new GameObject("Left YAM").AddComponent<YamArmRenderer>();
        left.transform.SetParent(root.transform, false);
        left.side = YamSide.Left;

        var right = new GameObject("Right YAM").AddComponent<YamArmRenderer>();
        right.transform.SetParent(root.transform, false);
        right.side = YamSide.Right;

        client.leftArm = left;
        client.rightArm = right;

        var leftHand = new GameObject("Left ORCA Hand").AddComponent<OrcaHandRenderer>();
        leftHand.transform.SetParent(root.transform, false);
        leftHand.side = YamSide.Left;
        client.leftHand = leftHand;

        var rightHand = new GameObject("Right ORCA Hand").AddComponent<OrcaHandRenderer>();
        rightHand.transform.SetParent(root.transform, false);
        rightHand.side = YamSide.Right;
        client.rightHand = rightHand;

        var op = new GameObject("Operator Vectors").AddComponent<OperatorVectorRenderer>();
        op.transform.SetParent(root.transform, false);
        client.operatorVectors = op;

        var hud = new GameObject("Status HUD").AddComponent<TeleopStatusHud>();
        hud.transform.SetParent(root.transform, false);
        client.statusHud = hud;

        EnsureSceneSupportObjects();
        return root;
    }

    public static void EnsureSceneSupportObjects()
    {
        Light light = GameObject.FindObjectOfType<Light>();
        GameObject lightObject = light != null ? light.gameObject : GameObject.Find("Key Light");
        if (lightObject == null)
        {
            lightObject = new GameObject("Key Light");
        }
        light = lightObject.GetComponent<Light>();
        if (light == null)
        {
            light = lightObject.AddComponent<Light>();
        }
        light.type = LightType.Directional;
        light.intensity = 1.2f;
        lightObject.transform.rotation = Quaternion.Euler(45f, -35f, 0f);

        Camera camera = Camera.main != null ? Camera.main : GameObject.FindObjectOfType<Camera>();
        GameObject cameraObject = camera != null ? camera.gameObject : GameObject.Find("Main Camera");
        if (cameraObject == null)
        {
            cameraObject = new GameObject("Main Camera");
        }
        camera = cameraObject.GetComponent<Camera>();
        if (camera == null)
        {
            camera = cameraObject.AddComponent<Camera>();
        }
        camera.tag = "MainCamera";
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.backgroundColor = new Color(0.035f, 0.04f, 0.05f);
        cameraObject.transform.position = new Vector3(0.0f, 1.35f, -1.8f);
        cameraObject.transform.rotation = Quaternion.Euler(22f, 0f, 0f);

        var grid = GameObject.Find("Floor");
        if (grid == null)
        {
            grid = GameObject.CreatePrimitive(PrimitiveType.Plane);
            grid.name = "Floor";
        }
        grid.transform.position = new Vector3(0f, 0f, 0.2f);
        grid.transform.localScale = new Vector3(1.5f, 1f, 1.5f);
        grid.GetComponent<Renderer>().material = TeleopUnityMaterials.Make(new Color(0.12f, 0.13f, 0.14f));
    }

    public static void ConfigureRuntime()
    {
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = 72;
        Screen.sleepTimeout = SleepTimeout.NeverSleep;
    }
}
