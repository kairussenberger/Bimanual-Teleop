using UnityEngine;

public sealed class OperatorVectorRenderer : MonoBehaviour
{
    public Vector3 overlayOrigin = new Vector3(-0.75f, 1.05f, 0.05f);
    public float scale = 0.8f;

    private const int ExpectedBodyVectorFloatCount = 3;
    private GameObject torso;
    private LineRenderer leftLine;
    private LineRenderer rightLine;
    private GameObject leftWrist;
    private GameObject rightWrist;
    private Material torsoMat;
    private Material leftMat;
    private Material rightMat;
    private bool initialized;

    private void Awake()
    {
        EnsureInitialized();
    }

    private void EnsureInitialized()
    {
        if (initialized)
        {
            return;
        }
        initialized = true;

        torsoMat = TeleopUnityMaterials.Make(new Color(0.95f, 0.85f, 0.25f));
        leftMat = TeleopUnityMaterials.Make(new Color(0.95f, 0.25f, 0.2f));
        rightMat = TeleopUnityMaterials.Make(new Color(0.2f, 0.55f, 1.0f));

        torso = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        torso.name = "Operator torso proxy";
        torso.transform.SetParent(transform, false);
        torso.transform.localScale = Vector3.one * 0.07f;
        torso.GetComponent<Renderer>().material = torsoMat;

        leftWrist = MakeWrist("Operator left wrist", leftMat);
        leftWrist.transform.SetParent(transform, false);
        rightWrist = MakeWrist("Operator right wrist", rightMat);
        rightWrist.transform.SetParent(transform, false);
        leftLine = MakeLine("Operator left torso-to-wrist", leftMat);
        rightLine = MakeLine("Operator right torso-to-wrist", rightMat);
    }

    public void Apply(RenderOperatorState state)
    {
        EnsureInitialized();

        bool hasState = state != null && state.hands != null;
        torso.SetActive(hasState);
        if (!hasState)
        {
            SetVisible(false, leftLine, leftWrist);
            SetVisible(false, rightLine, rightWrist);
            return;
        }

        torso.transform.position = overlayOrigin;
        ApplyHand(state.hands.left, leftLine, leftWrist);
        ApplyHand(state.hands.right, rightLine, rightWrist);
    }

    public Vector3 DebugLeftWristPosition()
    {
        return leftWrist.transform.position;
    }

    public Vector3 DebugRightWristPosition()
    {
        return rightWrist.transform.position;
    }

    public Vector3 DebugLeftLineEndPosition()
    {
        return leftLine.GetPosition(1);
    }

    public Vector3 DebugRightLineEndPosition()
    {
        return rightLine.GetPosition(1);
    }

    public bool DebugLeftWristActive()
    {
        return leftWrist.activeSelf;
    }

    public bool DebugRightWristActive()
    {
        return rightWrist.activeSelf;
    }

    public bool DebugLeftLineActive()
    {
        return leftLine.gameObject.activeSelf;
    }

    public bool DebugRightLineActive()
    {
        return rightLine.gameObject.activeSelf;
    }

    private void ApplyHand(RenderOperatorHand hand, LineRenderer line, GameObject wrist)
    {
        if (!ValidHand(hand))
        {
            SetVisible(false, line, wrist);
            return;
        }
        Vector3 target = overlayOrigin + TeleopUnityFrame.BodyVectorToUnity(hand.wrist_body) * scale;
        wrist.SetActive(true);
        wrist.transform.position = target;
        line.gameObject.SetActive(true);
        line.SetPosition(0, overlayOrigin);
        line.SetPosition(1, target);
    }

    private static bool ValidHand(RenderOperatorHand hand)
    {
        return hand != null
            && hand.tracked
            && FiniteArray(hand.wrist_body, ExpectedBodyVectorFloatCount);
    }

    private static bool FiniteArray(float[] values, int expectedLength)
    {
        if (values == null || values.Length != expectedLength)
        {
            return false;
        }
        for (int i = 0; i < values.Length; i++)
        {
            if (float.IsNaN(values[i]) || float.IsInfinity(values[i]))
            {
                return false;
            }
        }
        return true;
    }

    private static GameObject MakeWrist(string name, Material mat)
    {
        var obj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        obj.name = name;
        obj.transform.localScale = Vector3.one * 0.045f;
        obj.GetComponent<Renderer>().material = mat;
        return obj;
    }

    private LineRenderer MakeLine(string name, Material mat)
    {
        var obj = new GameObject(name);
        obj.transform.SetParent(transform, false);
        var line = obj.AddComponent<LineRenderer>();
        line.positionCount = 2;
        line.startWidth = 0.018f;
        line.endWidth = 0.018f;
        line.material = mat;
        line.useWorldSpace = true;
        return line;
    }

    private static void SetVisible(bool visible, LineRenderer line, GameObject wrist)
    {
        if (line != null) line.gameObject.SetActive(visible);
        if (wrist != null) wrist.SetActive(visible);
    }
}
