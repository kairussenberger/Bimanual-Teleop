using UnityEngine;

public static class TeleopUnityFrame
{
    public static Vector3 RobotWorldToUnity(Vector3 p)
    {
        return new Vector3(p.y, p.z, -p.x);
    }

    public static Quaternion RobotQuatToUnity(float[] q)
    {
        if (q == null || q.Length < 4)
        {
            return Quaternion.identity;
        }

        Quaternion robot = new Quaternion(q[1], q[2], q[3], q[0]);
        Vector3 right = RobotWorldToUnity(robot * Vector3.right);
        Vector3 up = RobotWorldToUnity(robot * Vector3.up);
        if (right.sqrMagnitude < 1e-6f || up.sqrMagnitude < 1e-6f)
        {
            return Quaternion.identity;
        }

        Vector3 forward = Vector3.Cross(right.normalized, up.normalized);
        return Quaternion.LookRotation(forward.normalized, up.normalized);
    }

    public static Vector3 BodyVectorToUnity(float[] body)
    {
        // body = [right, up, forward]. The operator overlay uses Unity x/y/z the same way.
        return new Vector3(body[0], body[1], body[2]);
    }
}
