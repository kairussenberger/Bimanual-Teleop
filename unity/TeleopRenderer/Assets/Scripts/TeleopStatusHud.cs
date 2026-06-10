using UnityEngine;

public sealed class TeleopStatusHud : MonoBehaviour
{
    public bool visible = true;
    public Rect panel = new Rect(16f, 16f, 560f, 148f);

    private string line0 = "TCP disconnected";
    private string line1 = "state=none";
    private string line2 = "hz=0.0 engaged=-- tracked=--";
    private string line3 = "calib=idle";
    private string line4 = "operator=head:none wrists=--";
    private string line5 = "cmd_err=--";
    private bool hasState;

    public void Apply(RenderState state, string connectionStatus, string endpoint, float now)
    {
        line0 = "TCP " + endpoint + " | " + Safe(connectionStatus);
        hasState = state != null;
        if (state == null || state.status == null)
        {
            line1 = "state=none";
            line2 = "hz=0.0 engaged=-- tracked=--";
            line3 = "calib=idle";
            line4 = "operator=head:none wrists=--";
            line5 = "cmd_err=--";
            return;
        }

        line1 = "state=ok updated=" + now.ToString("F2") + "s schema=" + state.v;
        line2 = "hz=" + state.status.hz.ToString("F1")
            + " engaged=" + Flags(state.status.engaged)
            + " tracked=" + Flags(state.status.tracked);
        line3 = CalibrationLine(state.status.calib);
        line4 = OperatorLine(state.op);
        line5 = CommandErrorLine(state.arms);
    }

    public void Clear(string connectionStatus, string endpoint, float now)
    {
        Apply(null, connectionStatus, endpoint, now);
    }

    public bool DebugHasState()
    {
        return hasState;
    }

    public string DebugLine(int index)
    {
        switch (index)
        {
            case 0: return line0;
            case 1: return line1;
            case 2: return line2;
            case 3: return line3;
            case 4: return line4;
            case 5: return line5;
            default: return "";
        }
    }

    private void OnGUI()
    {
        if (!visible)
        {
            return;
        }
        GUI.Box(panel, GUIContent.none);
        float x = panel.x + 10f;
        float y = panel.y + 8f;
        float w = panel.width - 20f;
        GUI.Label(new Rect(x, y, w, 20f), line0);
        GUI.Label(new Rect(x, y + 21f, w, 20f), line1);
        GUI.Label(new Rect(x, y + 42f, w, 20f), line2);
        GUI.Label(new Rect(x, y + 63f, w, 20f), line3);
        GUI.Label(new Rect(x, y + 84f, w, 20f), line4);
        GUI.Label(new Rect(x, y + 105f, w, 20f), line5);
    }

    private static string CalibrationLine(CalibrationStatus calib)
    {
        if (calib == null || !calib.active)
        {
            return "calib=idle";
        }
        float pct = Mathf.Clamp01(calib.progress) * 100.0f;
        return "calib=" + Safe(calib.phase) + " " + pct.ToString("F0") + "%"
            + " hold=" + calib.remaining.ToString("F1") + "s"
            + " sides=" + (calib.left ? "L" : "-") + (calib.right ? "R" : "-")
            + " " + Safe(calib.msg);
    }

    private static string OperatorLine(RenderOperatorState op)
    {
        if (op == null || op.hands == null)
        {
            return "operator=head:none wrists=--";
        }
        string head = op.head_pos != null && op.torso_pos != null ? "ok" : "none";
        return "operator=head:" + head
            + " wrists=" + (Tracked(op.hands.left) ? "L" : "-") + (Tracked(op.hands.right) ? "R" : "-");
    }

    private static string CommandErrorLine(RenderArms arms)
    {
        if (arms == null)
        {
            return "cmd_err=--";
        }
        return "cmd_err L=" + CommandErrorCm(arms.left) + " R=" + CommandErrorCm(arms.right);
    }

    private static string CommandErrorCm(RenderArmState arm)
    {
        if (arm == null || arm.ee_pos == null || arm.cmd_pos == null || arm.ee_pos.Length != 3 || arm.cmd_pos.Length != 3)
        {
            return "--";
        }
        float dx = arm.cmd_pos[0] - arm.ee_pos[0];
        float dy = arm.cmd_pos[1] - arm.ee_pos[1];
        float dz = arm.cmd_pos[2] - arm.ee_pos[2];
        float cm = Mathf.Sqrt(dx * dx + dy * dy + dz * dz) * 100.0f;
        return cm.ToString("F1") + "cm";
    }

    private static bool Tracked(RenderOperatorHand hand)
    {
        return hand != null && hand.tracked && hand.wrist_body != null && hand.wrist_body.Length == 3;
    }

    private static string Flags(SideFlags flags)
    {
        if (flags == null)
        {
            return "--";
        }
        return (flags.left ? "L" : "-") + (flags.right ? "R" : "-");
    }

    private static string Safe(string value)
    {
        return string.IsNullOrEmpty(value) ? "" : value;
    }
}
