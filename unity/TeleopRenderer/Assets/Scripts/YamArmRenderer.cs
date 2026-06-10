using UnityEngine;

public enum YamSide
{
    Left,
    Right
}

public sealed class YamArmRenderer : MonoBehaviour
{
    public YamSide side;

    private const int ExpectedLinkFloatCount = 24;
    private const float LinkRadius = 0.018f;
    private readonly GameObject[] joints = new GameObject[8];
    private readonly GameObject[] links = new GameObject[7];
    private GameObject eeMarker;
    private GameObject cmdMarker;
    private LineRenderer cmdErrorLine;
    private Material engagedMat;
    private Material idleMat;
    private Material lostMat;
    private Material eeMat;
    private Material cmdMat;
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

        engagedMat = TeleopUnityMaterials.Make(side == YamSide.Left ? new Color(0.95f, 0.25f, 0.2f) : new Color(0.2f, 0.55f, 1.0f));
        idleMat = TeleopUnityMaterials.Make(new Color(0.55f, 0.58f, 0.62f));
        lostMat = TeleopUnityMaterials.Make(new Color(0.18f, 0.18f, 0.2f));
        eeMat = TeleopUnityMaterials.Make(new Color(0.2f, 1.0f, 0.55f));
        cmdMat = TeleopUnityMaterials.Make(new Color(1.0f, 0.8f, 0.15f));

        for (int i = 0; i < joints.Length; i++)
        {
            joints[i] = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            joints[i].name = side + " joint " + i;
            joints[i].transform.SetParent(transform, false);
            joints[i].transform.localScale = Vector3.one * 0.045f;
        }

        for (int i = 0; i < links.Length; i++)
        {
            links[i] = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            links[i].name = side + " link " + i;
            links[i].transform.SetParent(transform, false);
        }

        eeMarker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        eeMarker.name = side + " achieved EE";
        eeMarker.transform.SetParent(transform, false);
        eeMarker.transform.localScale = Vector3.one * 0.07f;
        eeMarker.GetComponent<Renderer>().material = eeMat;

        cmdMarker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        cmdMarker.name = side + " commanded EE target";
        cmdMarker.transform.SetParent(transform, false);
        cmdMarker.transform.localScale = Vector3.one * 0.09f;
        cmdMarker.GetComponent<Renderer>().material = cmdMat;

        cmdErrorLine = MakeLine(side + " command error", cmdMat);
    }

    public void Apply(RenderArmState state, bool engaged, bool tracked)
    {
        EnsureInitialized();

        if (!ValidArmState(state))
        {
            SetVisible(false);
            return;
        }

        SetVisible(true);
        Material mat = tracked ? (engaged ? engagedMat : idleMat) : lostMat;
        foreach (var joint in joints)
        {
            joint.GetComponent<Renderer>().material = mat;
        }
        foreach (var link in links)
        {
            link.GetComponent<Renderer>().material = mat;
        }

        Vector3[] points = DecodeLinkPoints(state.link_pos);
        int jointCount = Mathf.Min(joints.Length, points.Length);
        for (int i = 0; i < jointCount; i++)
        {
            joints[i].transform.position = TeleopUnityFrame.RobotWorldToUnity(points[i]);
        }
        for (int i = jointCount; i < joints.Length; i++)
        {
            joints[i].SetActive(false);
        }

        int linkCount = Mathf.Min(links.Length, points.Length - 1);
        for (int i = 0; i < linkCount; i++)
        {
            PlaceCylinder(
                links[i].transform,
                TeleopUnityFrame.RobotWorldToUnity(points[i]),
                TeleopUnityFrame.RobotWorldToUnity(points[i + 1]),
                LinkRadius);
        }
        for (int i = linkCount; i < links.Length; i++)
        {
            links[i].SetActive(false);
        }

        if (state.ee_pos != null && state.ee_pos.Length >= 3)
        {
            eeMarker.transform.position = TeleopUnityFrame.RobotWorldToUnity(new Vector3(state.ee_pos[0], state.ee_pos[1], state.ee_pos[2]));
        }
        else if (points.Length > 0)
        {
            eeMarker.transform.position = TeleopUnityFrame.RobotWorldToUnity(points[points.Length - 1]);
        }
        else
        {
            eeMarker.SetActive(false);
        }

        if (state.cmd_pos != null)
        {
            cmdMarker.SetActive(true);
            cmdMarker.transform.position = TeleopUnityFrame.RobotWorldToUnity(new Vector3(state.cmd_pos[0], state.cmd_pos[1], state.cmd_pos[2]));
            cmdErrorLine.gameObject.SetActive(true);
            cmdErrorLine.SetPosition(0, eeMarker.transform.position);
            cmdErrorLine.SetPosition(1, cmdMarker.transform.position);
        }
        else
        {
            cmdMarker.SetActive(false);
            cmdErrorLine.gameObject.SetActive(false);
        }
    }

    public Vector3 DebugJointPosition(int index)
    {
        EnsureInitialized();
        return joints[index].transform.position;
    }

    public Vector3 DebugEePosition()
    {
        EnsureInitialized();
        return eeMarker.transform.position;
    }

    public Vector3 DebugCmdPosition()
    {
        EnsureInitialized();
        return cmdMarker.transform.position;
    }

    public bool DebugJointActive(int index)
    {
        EnsureInitialized();
        return joints[index].activeSelf;
    }

    public bool DebugCmdActive()
    {
        EnsureInitialized();
        return cmdMarker.activeSelf;
    }

    public Vector3 DebugCmdLineStartPosition()
    {
        EnsureInitialized();
        return cmdErrorLine.GetPosition(0);
    }

    public Vector3 DebugCmdLineEndPosition()
    {
        EnsureInitialized();
        return cmdErrorLine.GetPosition(1);
    }

    public bool DebugCmdLineActive()
    {
        EnsureInitialized();
        return cmdErrorLine.gameObject.activeSelf;
    }

    private static Vector3[] DecodeLinkPoints(float[] flat)
    {
        int n = flat.Length / 3;
        Vector3[] points = new Vector3[n];
        for (int i = 0; i < n; i++)
        {
            int k = i * 3;
            points[i] = new Vector3(flat[k], flat[k + 1], flat[k + 2]);
        }
        return points;
    }

    private static bool ValidArmState(RenderArmState state)
    {
        return state != null
            && FiniteArray(state.link_pos, ExpectedLinkFloatCount)
            && FiniteArray(state.ee_pos, 3)
            && (state.cmd_pos == null || FiniteArray(state.cmd_pos, 3));
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

    private LineRenderer MakeLine(string name, Material mat)
    {
        var obj = new GameObject(name);
        obj.transform.SetParent(transform, false);
        var line = obj.AddComponent<LineRenderer>();
        line.positionCount = 2;
        line.startWidth = 0.012f;
        line.endWidth = 0.012f;
        line.material = mat;
        line.useWorldSpace = true;
        return line;
    }

    private void SetVisible(bool visible)
    {
        foreach (var joint in joints)
        {
            if (joint != null) joint.SetActive(visible);
        }
        foreach (var link in links)
        {
            if (link != null) link.SetActive(visible);
        }
        if (eeMarker != null) eeMarker.SetActive(visible);
        if (cmdMarker != null) cmdMarker.SetActive(visible);
        if (cmdErrorLine != null) cmdErrorLine.gameObject.SetActive(visible);
    }
}
