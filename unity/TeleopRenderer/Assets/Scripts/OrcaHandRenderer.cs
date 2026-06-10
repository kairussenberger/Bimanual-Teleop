using System;
using UnityEngine;

public sealed class OrcaHandRenderer : MonoBehaviour
{
    public YamSide side;

    private const int ExpectedJointCount = 17;
    private const int FingerCount = 5;
    private const int SegmentsPerFinger = 3;
    private const float SegmentRadius = 0.011f;
    private const float PalmWidth = 0.105f;
    private const float PalmLength = 0.075f;

    private readonly GameObject[,] segments = new GameObject[FingerCount, SegmentsPerFinger];
    private readonly GameObject[,] joints = new GameObject[FingerCount, SegmentsPerFinger + 1];
    private GameObject palm;
    private Material trackedMat;
    private Material lostMat;
    private Material palmMat;
    private bool initialized;

    private static readonly string[] Fingers = {"thumb", "index", "middle", "ring", "pinky"};
    private static readonly float[] FingerX = {-0.052f, -0.030f, -0.010f, 0.012f, 0.034f};
    private static readonly float[] FingerLengths = {0.030f, 0.040f, 0.044f, 0.041f, 0.035f};

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

        trackedMat = TeleopUnityMaterials.Make(side == YamSide.Left ? new Color(0.95f, 0.35f, 0.28f) : new Color(0.32f, 0.65f, 1.0f));
        lostMat = TeleopUnityMaterials.Make(new Color(0.2f, 0.2f, 0.22f));
        palmMat = TeleopUnityMaterials.Make(new Color(0.78f, 0.78f, 0.72f));

        palm = GameObject.CreatePrimitive(PrimitiveType.Cube);
        palm.name = side + " ORCA palm";
        palm.transform.SetParent(transform, false);
        palm.transform.localScale = new Vector3(PalmWidth, 0.026f, PalmLength);
        palm.GetComponent<Renderer>().material = palmMat;

        for (int f = 0; f < FingerCount; f++)
        {
            for (int j = 0; j < SegmentsPerFinger + 1; j++)
            {
                joints[f, j] = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                joints[f, j].name = side + " " + Fingers[f] + " joint " + j;
                joints[f, j].transform.SetParent(transform, false);
                joints[f, j].transform.localScale = Vector3.one * 0.026f;
            }
            for (int s = 0; s < SegmentsPerFinger; s++)
            {
                segments[f, s] = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                segments[f, s].name = side + " " + Fingers[f] + " segment " + s;
                segments[f, s].transform.SetParent(transform, false);
            }
        }
    }

    public void Apply(RenderHandState hand, RenderArmState arm, bool tracked)
    {
        EnsureInitialized();

        if (!ValidHandState(hand) || arm == null || !FiniteArray(arm.ee_pos, 3) || !FiniteArray(arm.ee_quat, 4))
        {
            SetVisible(false);
            return;
        }

        SetVisible(true);
        Material mat = tracked ? trackedMat : lostMat;
        palm.GetComponent<Renderer>().material = tracked ? palmMat : lostMat;
        foreach (GameObject obj in joints)
        {
            obj.GetComponent<Renderer>().material = mat;
        }
        foreach (GameObject obj in segments)
        {
            obj.GetComponent<Renderer>().material = mat;
        }

        Vector3 palmPos = TeleopUnityFrame.RobotWorldToUnity(new Vector3(arm.ee_pos[0], arm.ee_pos[1], arm.ee_pos[2]));
        Quaternion palmRot = TeleopUnityFrame.RobotQuatToUnity(arm.ee_quat);
        palm.transform.position = palmPos;
        palm.transform.rotation = palmRot;

        for (int f = 0; f < FingerCount; f++)
        {
            DrawFinger(f, hand, palmPos, palmRot);
        }
    }

    public Vector3 DebugPalmPosition()
    {
        EnsureInitialized();
        return palm.transform.position;
    }

    public Quaternion DebugPalmRotation()
    {
        EnsureInitialized();
        return palm.transform.rotation;
    }

    public bool DebugPalmActive()
    {
        EnsureInitialized();
        return palm.activeSelf;
    }

    private void DrawFinger(int finger, RenderHandState hand, Vector3 palmPos, Quaternion palmRot)
    {
        string name = Fingers[finger];
        float abd = Value(hand, name + "_abd");
        float mcp = Value(hand, name + "_mcp");
        float pip = Value(hand, name + "_pip");
        if (name == "thumb")
        {
            abd = Value(hand, "thumb_abd");
            mcp = 0.55f * Value(hand, "thumb_mcp") + 0.25f * Value(hand, "thumb_cmc");
            pip = Value(hand, "thumb_dip");
        }

        float sideSign = side == YamSide.Left ? -1.0f : 1.0f;
        float baseX = FingerX[finger] * sideSign;
        float baseZ = name == "thumb" ? -0.020f : 0.020f;
        float spread = Mathf.Clamp(abd, -70.0f, 70.0f) * Mathf.Deg2Rad * 0.45f;
        float[] curls = {
            Mathf.Clamp(mcp, -20.0f, 100.0f) * Mathf.Deg2Rad,
            Mathf.Clamp(pip, 0.0f, 110.0f) * Mathf.Deg2Rad,
            Mathf.Clamp(pip, 0.0f, 110.0f) * Mathf.Deg2Rad * 0.75f
        };

        Vector3[] pts = new Vector3[SegmentsPerFinger + 1];
        pts[0] = palmPos + palmRot * new Vector3(baseX, 0.0f, baseZ);
        Vector3 dir = name == "thumb"
            ? new Vector3(0.45f * sideSign, 0.0f, 0.90f).normalized
            : new Vector3(Mathf.Sin(spread) * sideSign, 0.0f, Mathf.Cos(spread)).normalized;
        float curlAccum = 0.0f;
        float len = FingerLengths[finger];
        for (int i = 0; i < SegmentsPerFinger; i++)
        {
            curlAccum += curls[i];
            Vector3 bent = new Vector3(dir.x, -Mathf.Sin(curlAccum), dir.z * Mathf.Cos(curlAccum)).normalized;
            pts[i + 1] = pts[i] + palmRot * (bent * len);
        }

        for (int i = 0; i < pts.Length; i++)
        {
            joints[finger, i].transform.position = pts[i];
        }
        for (int i = 0; i < SegmentsPerFinger; i++)
        {
            PlaceCylinder(segments[finger, i].transform, pts[i], pts[i + 1], SegmentRadius);
        }
    }

    private static float Value(RenderHandState hand, string name)
    {
        if (hand.names == null || hand.q == null)
        {
            return 0.0f;
        }
        int n = Math.Min(hand.names.Length, hand.q.Length);
        for (int i = 0; i < n; i++)
        {
            if (hand.names[i] == name)
            {
                return hand.q[i];
            }
        }
        return 0.0f;
    }

    private static bool ValidHandState(RenderHandState hand)
    {
        return hand != null
            && hand.names != null && hand.names.Length == ExpectedJointCount
            && FiniteArray(hand.q, ExpectedJointCount);
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

    private static void PlaceCylinder(Transform t, Vector3 a, Vector3 b, float radius)
    {
        Vector3 d = b - a;
        float len = d.magnitude;
        if (len < 0.0001f)
        {
            t.gameObject.SetActive(false);
            return;
        }
        t.gameObject.SetActive(true);
        t.position = (a + b) * 0.5f;
        t.rotation = Quaternion.FromToRotation(Vector3.up, d.normalized);
        t.localScale = new Vector3(radius, len * 0.5f, radius);
    }

    private void SetVisible(bool visible)
    {
        if (palm != null) palm.SetActive(visible);
        foreach (GameObject obj in joints)
        {
            if (obj != null) obj.SetActive(visible);
        }
        foreach (GameObject obj in segments)
        {
            if (obj != null) obj.SetActive(visible);
        }
    }
}
