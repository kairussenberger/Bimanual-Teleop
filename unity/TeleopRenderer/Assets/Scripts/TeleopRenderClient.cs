using System;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

[Serializable]
public class RenderState
{
    public int v;
    public float stamp;
    public RenderArms arms;
    public RenderHands hand_render;
    public RenderOperatorState op;
    public RenderStatus status;
}

[Serializable]
public class RenderArms
{
    public RenderArmState left;
    public RenderArmState right;
}

[Serializable]
public class RenderArmState
{
    public float[] q;
    public float[] link_pos;
    public float[] ee_pos;
    public float[] ee_quat;
    public float[] cmd_pos;
    public float[] cmd_quat;
    public float[] margins;
}

[Serializable]
public class RenderHands
{
    public RenderHandState left;
    public RenderHandState right;
}

[Serializable]
public class RenderHandState
{
    public string[] names;
    public float[] q;
}

[Serializable]
public class RenderOperatorState
{
    public float[] torso_from_head;
    public float[] head_pos;
    public float[] torso_pos;
    public RenderOperatorHands hands;
}

[Serializable]
public class RenderOperatorHands
{
    public RenderOperatorHand left;
    public RenderOperatorHand right;
}

[Serializable]
public class RenderOperatorHand
{
    public bool tracked;
    public float[] wrist_body;
    public float[] raw_wrist;
}

[Serializable]
public class RenderStatus
{
    public SideFlags engaged;
    public SideFlags tracked;
    public CalibrationStatus calib;
    public float hz;
}

[Serializable]
public class SideFlags
{
    public bool left;
    public bool right;
}

[Serializable]
public class CalibrationStatus
{
    public bool active;
    public string phase;
    public float progress;
    public float remaining;
    public bool left;
    public bool right;
    public string msg;
}

public sealed class TeleopRenderClient : MonoBehaviour
{
    private const int ExpectedSchemaVersion = 2;
    private const int ExpectedArmJointCount = 6;
    private const int ExpectedArmLinkFloatCount = 24;
    private const int ExpectedHandJointCount = 17;
    private const int ExpectedVec3FloatCount = 3;
    private const int ExpectedQuatFloatCount = 4;

    public string host = "127.0.0.1";
    public int port = 8102;
    public int connectTimeoutMs = 500;
    public float stateTimeoutSeconds = 0.75f;
    public YamArmRenderer leftArm;
    public YamArmRenderer rightArm;
    public OrcaHandRenderer leftHand;
    public OrcaHandRenderer rightHand;
    public OperatorVectorRenderer operatorVectors;
    public TeleopStatusHud statusHud;

    private Thread worker;
    private volatile bool running;
    private readonly object gate = new object();
    private readonly object socketGate = new object();
    private string latestJson;
    private RenderState latestState;
    private string status = "disconnected";
    private TcpClient activeClient;
    private float lastValidStateTime = -1.0f;
    private bool renderersHidden = true;

    private void OnEnable()
    {
        if (!Application.isPlaying)
        {
            return;
        }
        running = true;
        worker = new Thread(ReadLoop) { IsBackground = true };
        worker.Start();
    }

    private void OnDisable()
    {
        running = false;
        CloseActiveClient();
        if (worker != null && worker.IsAlive)
        {
            worker.Join(200);
        }
        worker = null;
    }

    private void Update()
    {
        float now = Time.time;
        string json = null;
        lock (gate)
        {
            if (!string.IsNullOrEmpty(latestJson))
            {
                json = latestJson;
                latestJson = null;
            }
        }

        if (json == null)
        {
            HideIfStale(now);
            UpdateStatusHud(now);
            return;
        }

        ApplyJsonAt(json, now);
    }

    public void ApplyState(RenderState state)
    {
        ApplyStateAt(state, Time.time);
    }

    public void ApplyJsonAt(string json, float now)
    {
        try
        {
            RenderState parsed = JsonUtility.FromJson<RenderState>(json);
            ApplyStateAt(parsed, now);
        }
        catch (Exception e)
        {
            HideRenderers();
            status = "json error: " + e.Message;
            UpdateStatusHud(now);
        }
    }

    public void ApplyStateAt(RenderState state, float now)
    {
        if (!ValidStateShape(state))
        {
            HideRenderers();
            status = SupportedSchema(state) ? "invalid state" : "schema mismatch";
            UpdateStatusHud(now);
            return;
        }

        lastValidStateTime = now;
        renderersHidden = false;
        latestState = state;
        status = "rendering";
        RenderArmState leftArmState = state != null && state.arms != null ? state.arms.left : null;
        RenderArmState rightArmState = state != null && state.arms != null ? state.arms.right : null;
        RenderHandState leftHandState = state != null && state.hand_render != null ? state.hand_render.left : null;
        RenderHandState rightHandState = state != null && state.hand_render != null ? state.hand_render.right : null;
        SideFlags engaged = state != null && state.status != null ? state.status.engaged : null;
        SideFlags tracked = state != null && state.status != null ? state.status.tracked : null;

        if (leftArm != null)
        {
            leftArm.Apply(leftArmState, Flag(engaged, YamSide.Left), Flag(tracked, YamSide.Left));
        }
        if (rightArm != null)
        {
            rightArm.Apply(rightArmState, Flag(engaged, YamSide.Right), Flag(tracked, YamSide.Right));
        }
        if (leftHand != null)
        {
            leftHand.Apply(leftHandState, leftArmState, Flag(tracked, YamSide.Left));
        }
        if (rightHand != null)
        {
            rightHand.Apply(rightHandState, rightArmState, Flag(tracked, YamSide.Right));
        }
        if (operatorVectors != null)
        {
            operatorVectors.Apply(state != null ? state.op : null);
        }
        UpdateStatusHud(now);
    }

    public void HideIfStale(float now)
    {
        if (renderersHidden || stateTimeoutSeconds <= 0.0f || lastValidStateTime < 0.0f)
        {
            return;
        }
        if (now - lastValidStateTime > stateTimeoutSeconds)
        {
            HideRenderers();
            latestState = null;
            status = "stale";
            UpdateStatusHud(now);
        }
    }

    private void HideRenderers()
    {
        if (leftArm != null) leftArm.Apply(null, false, false);
        if (rightArm != null) rightArm.Apply(null, false, false);
        if (leftHand != null) leftHand.Apply(null, null, false);
        if (rightHand != null) rightHand.Apply(null, null, false);
        if (operatorVectors != null) operatorVectors.Apply(null);
        latestState = null;
        renderersHidden = true;
    }

    public bool DebugHasLatestState()
    {
        return latestState != null;
    }

    private void UpdateStatusHud(float now)
    {
        if (statusHud == null)
        {
            return;
        }
        if (latestState != null)
        {
            statusHud.Apply(latestState, status, host + ":" + port, now);
        }
        else
        {
            statusHud.Clear(status, host + ":" + port, now);
        }
    }

    private static bool SupportedSchema(RenderState state)
    {
        return state != null && state.v == ExpectedSchemaVersion;
    }

    private static bool ValidStateShape(RenderState state)
    {
        return SupportedSchema(state)
            && state.arms != null
            && state.arms.left != null
            && state.arms.right != null
            && ValidArmStateShape(state.arms.left)
            && ValidArmStateShape(state.arms.right)
            && state.hand_render != null
            && state.hand_render.left != null
            && state.hand_render.right != null
            && ValidHandRenderShape(state.hand_render.left)
            && ValidHandRenderShape(state.hand_render.right)
            && state.op != null
            && state.op.hands != null
            && state.op.hands.left != null
            && state.op.hands.right != null
            && ValidOperatorStateShape(state.op)
            && state.status != null
            && state.status.engaged != null
            && state.status.tracked != null
            && ValidStatusShape(state.status);
    }

    private static bool ValidArmStateShape(RenderArmState arm)
    {
        return arm != null
            && FiniteArray(arm.q, ExpectedArmJointCount)
            && FiniteArray(arm.link_pos, ExpectedArmLinkFloatCount)
            && FiniteArray(arm.ee_pos, ExpectedVec3FloatCount)
            && FiniteArray(arm.ee_quat, ExpectedQuatFloatCount)
            && (arm.cmd_pos == null || FiniteArray(arm.cmd_pos, ExpectedVec3FloatCount))
            && (arm.cmd_quat == null || FiniteArray(arm.cmd_quat, ExpectedQuatFloatCount))
            && FiniteArray(arm.margins, ExpectedArmJointCount);
    }

    private static bool ValidHandRenderShape(RenderHandState hand)
    {
        return hand != null
            && hand.names != null && hand.names.Length == ExpectedHandJointCount
            && FiniteArray(hand.q, ExpectedHandJointCount);
    }

    private static bool ValidOperatorStateShape(RenderOperatorState op)
    {
        return op != null
            && Vec3(op.torso_from_head)
            && NullableVec3(op.head_pos)
            && NullableVec3(op.torso_pos)
            && ((op.head_pos == null && op.torso_pos == null) || (op.head_pos != null && op.torso_pos != null))
            && op.hands != null
            && ValidOperatorHandShape(op.hands.left)
            && ValidOperatorHandShape(op.hands.right);
    }

    private static bool ValidOperatorHandShape(RenderOperatorHand hand)
    {
        return hand != null
            && (hand.tracked ? Vec3(hand.wrist_body) : hand.wrist_body == null)
            && NullableVec3(hand.raw_wrist);
    }

    private static bool ValidStatusShape(RenderStatus status)
    {
        return status != null
            && FiniteValue(status.hz)
            && (status.calib == null || ValidCalibrationShape(status.calib));
    }

    private static bool ValidCalibrationShape(CalibrationStatus calib)
    {
        return calib != null
            && FiniteValue(calib.progress)
            && FiniteValue(calib.remaining);
    }

    private static bool Vec3(float[] values)
    {
        return FiniteArray(values, ExpectedVec3FloatCount);
    }

    private static bool NullableVec3(float[] values)
    {
        return values == null || FiniteArray(values, ExpectedVec3FloatCount);
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

    private static bool FiniteValue(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }

    private static bool Flag(SideFlags flags, YamSide side)
    {
        if (flags == null)
        {
            return false;
        }
        return side == YamSide.Left ? flags.left : flags.right;
    }

    private void ReadLoop()
    {
        while (running)
        {
            try
            {
                using (var client = new TcpClient())
                {
                    client.NoDelay = true;
                    SetActiveClient(client);
                    if (!TryConnect(client))
                    {
                        throw new TimeoutException("connect timeout");
                    }
                    status = "connected";
                    using (var reader = new StreamReader(client.GetStream()))
                    {
                        while (running)
                        {
                            string line = reader.ReadLine();
                            if (line == null)
                            {
                                break;
                            }
                            lock (gate)
                            {
                                latestJson = line;
                            }
                        }
                    }
                }
            }
            catch (Exception e)
            {
                if (running)
                {
                    status = "waiting: " + e.GetType().Name;
                    Thread.Sleep(500);
                }
            }
            finally
            {
                SetActiveClient(null);
            }
        }
    }

    private bool TryConnect(TcpClient client)
    {
        IAsyncResult result = client.BeginConnect(host, port, null, null);
        bool ok = result.AsyncWaitHandle.WaitOne(connectTimeoutMs);
        if (!ok)
        {
            client.Close();
            return false;
        }
        client.EndConnect(result);
        return true;
    }

    private void SetActiveClient(TcpClient client)
    {
        lock (socketGate)
        {
            activeClient = client;
        }
    }

    private void CloseActiveClient()
    {
        lock (socketGate)
        {
            if (activeClient != null)
            {
                activeClient.Close();
                activeClient = null;
            }
        }
    }
}
