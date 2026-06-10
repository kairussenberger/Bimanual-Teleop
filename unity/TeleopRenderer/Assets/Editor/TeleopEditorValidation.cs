#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEngine;

public static class TeleopEditorValidation
{
    [MenuItem("Teleop/Run Renderer Validation")]
    public static void RunFromMenu()
    {
        Run();
    }

    public static void Run()
    {
        RenderState sample = ValidateJsonUtilityDto();
        ValidateCoordinateFrame();
        ValidateMaterialFactory();
        ValidateBootstrapRuntimeConfig();
        ValidateBootstrapWiring();
        ValidateRendererApplyPath(sample);
        ValidateSceneAssetAndBuildSettings();
        Debug.Log("TeleopRenderer editor validation passed");
    }

    private static RenderState ValidateJsonUtilityDto()
    {
        string json = File.ReadAllText(SamplePath());
        RenderState state = JsonUtility.FromJson<RenderState>(json);
        Require(state != null, "JsonUtility returned null");
        Require(state.arms != null && state.arms.left != null && state.arms.right != null, "arms DTO missing");
        Require(state.arms.left.link_pos != null && state.arms.left.link_pos.Length == 24, "left link_pos shape");
        Require(state.arms.right.link_pos != null && state.arms.right.link_pos.Length == 24, "right link_pos shape");
        Require(state.hand_render != null && state.hand_render.left != null && state.hand_render.right != null, "hand_render DTO missing");
        Require(state.hand_render.left.names.Length == 17 && state.hand_render.left.q.Length == 17, "left hand_render shape");
        Require(state.hand_render.right.names.Length == 17 && state.hand_render.right.q.Length == 17, "right hand_render shape");
        Require(state.op != null && state.op.hands != null, "operator DTO missing");
        Require(state.op.hands.left.wrist_body.Length == 3, "left operator wrist_body shape");
        Require(state.op.hands.right.wrist_body.Length == 3, "right operator wrist_body shape");
        return state;
    }

    private static void ValidateBootstrapWiring()
    {
        DestroyIfPresent("Main Camera");
        DestroyIfPresent("Key Light");
        DestroyIfPresent("Floor");
        GameObject root = TeleopSceneBootstrap.CreateRendererRoot();
        try
        {
            Require(root.name == "TeleopRenderer", "bootstrap root name");
            var client = root.GetComponent<TeleopRenderClient>();
            Require(client != null, "bootstrap client missing");
            Require(client.host == "127.0.0.1" && client.port == 8102, "client endpoint defaults");
            Require(client.leftArm != null && client.leftArm.side == YamSide.Left, "left arm not wired");
            Require(client.rightArm != null && client.rightArm.side == YamSide.Right, "right arm not wired");
            Require(client.leftHand != null && client.leftHand.side == YamSide.Left, "left hand not wired");
            Require(client.rightHand != null && client.rightHand.side == YamSide.Right, "right hand not wired");
            Require(client.operatorVectors != null, "operator overlay not wired");
            Require(client.statusHud != null, "status HUD not wired");
            Require(GameObject.Find("Main Camera") != null, "main camera missing");
            Require(GameObject.Find("Key Light") != null, "key light missing");
            Require(GameObject.Find("Floor") != null, "floor missing");
        }
        finally
        {
            Object.DestroyImmediate(root);
            DestroyIfPresent("Main Camera");
            DestroyIfPresent("Key Light");
            DestroyIfPresent("Floor");
        }

        var existingCamera = new GameObject("Main Camera");
        existingCamera.AddComponent<Camera>();
        existingCamera.tag = "MainCamera";
        var existingLight = new GameObject("Key Light");
        existingLight.AddComponent<Light>();
        var existingFloor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        existingFloor.name = "Floor";
        root = TeleopSceneBootstrap.CreateRendererRoot();
        try
        {
            Require(GameObject.FindObjectsOfType<Camera>().Length == 1, "bootstrap duplicated existing camera");
            Require(GameObject.FindObjectsOfType<Light>().Length == 1, "bootstrap duplicated existing light");
            Require(GameObject.FindGameObjectsWithTag("MainCamera").Length == 1, "bootstrap duplicated MainCamera tag");
            Require(GameObject.Find("Floor") == existingFloor, "bootstrap duplicated existing floor");
        }
        finally
        {
            Object.DestroyImmediate(root);
            DestroyIfPresent("Main Camera");
            DestroyIfPresent("Key Light");
            DestroyIfPresent("Floor");
        }
    }

    private static void ValidateCoordinateFrame()
    {
        Vector3 pos = TeleopUnityFrame.RobotWorldToUnity(new Vector3(-1.0f, 2.0f, 3.0f));
        Require(VectorClose(pos, new Vector3(2.0f, 3.0f, 1.0f)), "robot world position conversion");

        Quaternion rot = TeleopUnityFrame.RobotQuatToUnity(new float[] {1.0f, 0.0f, 0.0f, 0.0f});
        Require(VectorClose(rot * Vector3.right, new Vector3(0.0f, 0.0f, -1.0f)), "robot identity right axis conversion");
        Require(VectorClose(rot * Vector3.up, new Vector3(1.0f, 0.0f, 0.0f)), "robot identity up axis conversion");

        Vector3 body = TeleopUnityFrame.BodyVectorToUnity(new float[] {0.2f, 0.3f, 0.4f});
        Require(VectorClose(body, new Vector3(0.2f, 0.3f, 0.4f)), "operator body vector conversion");
    }

    private static void ValidateMaterialFactory()
    {
        Color expected = new Color(0.25f, 0.5f, 0.75f, 1.0f);
        Material material = TeleopUnityMaterials.Make(expected);
        Require(material != null, "material factory returned null");
        Require(material.shader != null, "material factory returned material without shader");
        Require(ColorClose(material.color, expected), "material factory did not preserve color");
        Object.DestroyImmediate(material);
    }

    private static void ValidateBootstrapRuntimeConfig()
    {
        int oldVSync = QualitySettings.vSyncCount;
        int oldTargetFrameRate = Application.targetFrameRate;
        int oldSleepTimeout = Screen.sleepTimeout;
        try
        {
            TeleopSceneBootstrap.ConfigureRuntime();
            Require(QualitySettings.vSyncCount == 0, "bootstrap runtime did not disable vSync");
            Require(Application.targetFrameRate == 72, "bootstrap runtime did not target 72 FPS");
            Require(Screen.sleepTimeout == SleepTimeout.NeverSleep, "bootstrap runtime did not prevent sleep");
        }
        finally
        {
            QualitySettings.vSyncCount = oldVSync;
            Application.targetFrameRate = oldTargetFrameRate;
            Screen.sleepTimeout = oldSleepTimeout;
        }
    }

    private static void ValidateRendererApplyPath(RenderState sample)
    {
        var root = new GameObject("TeleopEditorValidation");
        try
        {
            var leftArm = new GameObject("LeftArm").AddComponent<YamArmRenderer>();
            leftArm.transform.SetParent(root.transform, false);
            leftArm.side = YamSide.Left;

            var rightArm = new GameObject("RightArm").AddComponent<YamArmRenderer>();
            rightArm.transform.SetParent(root.transform, false);
            rightArm.side = YamSide.Right;

            var leftHand = new GameObject("LeftHand").AddComponent<OrcaHandRenderer>();
            leftHand.transform.SetParent(root.transform, false);
            leftHand.side = YamSide.Left;

            var rightHand = new GameObject("RightHand").AddComponent<OrcaHandRenderer>();
            rightHand.transform.SetParent(root.transform, false);
            rightHand.side = YamSide.Right;

            var overlay = new GameObject("OperatorOverlay").AddComponent<OperatorVectorRenderer>();
            overlay.transform.SetParent(root.transform, false);

            var hud = new GameObject("StatusHud").AddComponent<TeleopStatusHud>();
            hud.transform.SetParent(root.transform, false);

            leftArm.Apply(sample.arms.left, true, true);
            rightArm.Apply(sample.arms.right, true, true);
            leftHand.Apply(sample.hand_render.left, sample.arms.left, true);
            rightHand.Apply(sample.hand_render.right, sample.arms.right, true);
            overlay.Apply(sample.op);

            Require(root.GetComponentsInChildren<Renderer>(true).Length > 20, "renderer primitives were not created");
            Require(root.GetComponentsInChildren<LineRenderer>(true).Length == 4, "operator/command vector lines missing");

            Vector3 expectedLeftBase = TeleopUnityFrame.RobotWorldToUnity(VectorFrom(sample.arms.left.link_pos, 0));
            Vector3 expectedRightBase = TeleopUnityFrame.RobotWorldToUnity(VectorFrom(sample.arms.right.link_pos, 0));
            Vector3 expectedLeftEe = TeleopUnityFrame.RobotWorldToUnity(VectorFrom(sample.arms.left.ee_pos, 0));
            Vector3 expectedRightEe = TeleopUnityFrame.RobotWorldToUnity(VectorFrom(sample.arms.right.ee_pos, 0));
            Vector3 expectedLeftCmd = TeleopUnityFrame.RobotWorldToUnity(VectorFrom(sample.arms.left.cmd_pos, 0));
            Vector3 expectedRightCmd = TeleopUnityFrame.RobotWorldToUnity(VectorFrom(sample.arms.right.cmd_pos, 0));
            Vector3 expectedLeftWrist = overlay.overlayOrigin
                + TeleopUnityFrame.BodyVectorToUnity(sample.op.hands.left.wrist_body) * overlay.scale;
            Vector3 expectedRightWrist = overlay.overlayOrigin
                + TeleopUnityFrame.BodyVectorToUnity(sample.op.hands.right.wrist_body) * overlay.scale;

            Require(VectorClose(leftArm.DebugJointPosition(0), expectedLeftBase), "left arm did not apply link_pos[0]");
            Require(VectorClose(rightArm.DebugJointPosition(0), expectedRightBase), "right arm did not apply link_pos[0]");
            Require(VectorClose(leftArm.DebugEePosition(), expectedLeftEe), "left arm did not apply ee_pos");
            Require(VectorClose(rightArm.DebugEePosition(), expectedRightEe), "right arm did not apply ee_pos");
            Require(VectorClose(leftArm.DebugCmdPosition(), expectedLeftCmd) && leftArm.DebugCmdActive(), "left arm did not apply cmd_pos");
            Require(VectorClose(rightArm.DebugCmdPosition(), expectedRightCmd) && rightArm.DebugCmdActive(), "right arm did not apply cmd_pos");
            Require(leftArm.DebugCmdLineActive()
                && VectorClose(leftArm.DebugCmdLineStartPosition(), expectedLeftEe)
                && VectorClose(leftArm.DebugCmdLineEndPosition(), expectedLeftCmd),
                "left arm did not draw command error line");
            Require(rightArm.DebugCmdLineActive()
                && VectorClose(rightArm.DebugCmdLineStartPosition(), expectedRightEe)
                && VectorClose(rightArm.DebugCmdLineEndPosition(), expectedRightCmd),
                "right arm did not draw command error line");
            Require(VectorClose(leftHand.DebugPalmPosition(), expectedLeftEe), "left hand did not anchor to achieved EE");
            Require(VectorClose(rightHand.DebugPalmPosition(), expectedRightEe), "right hand did not anchor to achieved EE");
            Require(VectorClose(overlay.DebugLeftWristPosition(), expectedLeftWrist), "operator overlay did not apply wrist_body");
            Require(VectorClose(overlay.DebugRightWristPosition(), expectedRightWrist), "operator overlay did not apply right wrist_body");
            Require(VectorClose(overlay.DebugLeftLineEndPosition(), expectedLeftWrist), "operator overlay line did not end at wrist_body");
            Require(VectorClose(overlay.DebugRightLineEndPosition(), expectedRightWrist), "operator overlay right line did not end at wrist_body");
            Require(overlay.DebugLeftWristActive() && overlay.DebugLeftLineActive(), "operator overlay did not show tracked left wrist");

            overlay.Apply(new RenderOperatorState {
                torso_from_head = sample.op.torso_from_head,
                head_pos = null,
                torso_pos = null,
                hands = new RenderOperatorHands {
                    left = new RenderOperatorHand {tracked = false, wrist_body = null, raw_wrist = sample.op.hands.left.raw_wrist},
                    right = new RenderOperatorHand {tracked = true, wrist_body = null, raw_wrist = sample.op.hands.right.raw_wrist}
                }
            });
            Require(!overlay.DebugLeftWristActive() && !overlay.DebugLeftLineActive(), "operator overlay left wrist_body null/untracked did not hide");
            Require(!overlay.DebugRightWristActive() && !overlay.DebugRightLineActive(), "operator overlay right wrist_body null/tracked did not hide");

            overlay.Apply(new RenderOperatorState {
                torso_from_head = sample.op.torso_from_head,
                head_pos = sample.op.head_pos,
                torso_pos = sample.op.torso_pos,
                hands = new RenderOperatorHands {
                    left = new RenderOperatorHand {tracked = true, wrist_body = new float[] {0.1f, 0.2f, 0.3f, 0.4f}, raw_wrist = sample.op.hands.left.raw_wrist},
                    right = sample.op.hands.right
                }
            });
            Require(!overlay.DebugLeftWristActive() && !overlay.DebugLeftLineActive(), "operator overlay malformed wrist_body did not hide left");
            Require(overlay.DebugRightWristActive() && overlay.DebugRightLineActive(), "operator overlay malformed left wrist_body hid right");

            overlay.Apply(new RenderOperatorState {
                torso_from_head = sample.op.torso_from_head,
                head_pos = sample.op.head_pos,
                torso_pos = sample.op.torso_pos,
                hands = new RenderOperatorHands {
                    left = sample.op.hands.left,
                    right = new RenderOperatorHand {tracked = true, wrist_body = new float[] {0.1f, 0.2f, 0.3f, 0.4f}, raw_wrist = sample.op.hands.right.raw_wrist}
                }
            });
            Require(overlay.DebugLeftWristActive() && overlay.DebugLeftLineActive(), "operator overlay malformed right wrist_body hid left");
            Require(!overlay.DebugRightWristActive() && !overlay.DebugRightLineActive(), "operator overlay malformed wrist_body did not hide right");

            var client = root.AddComponent<TeleopRenderClient>();
            client.leftArm = leftArm;
            client.rightArm = rightArm;
            client.leftHand = leftHand;
            client.rightHand = rightHand;
            client.operatorVectors = overlay;
            client.statusHud = hud;
            client.ApplyState(sample);
            client.ApplyStateAt(sample, 10.0f);
            Require(client.DebugHasLatestState() && leftArm.DebugJointActive(0) && leftHand.DebugPalmActive() && overlay.DebugLeftWristActive(),
                "client did not show valid state before stale timeout");
            Require(hud.DebugHasState() && hud.DebugLine(0).Contains("rendering") && hud.DebugLine(2).Contains("engaged="),
                "status HUD did not show accepted render state");
            Require(hud.DebugLine(5).Contains("cmd_err L=") && hud.DebugLine(5).Contains("cm"),
                "status HUD did not show command error");
            client.HideIfStale(10.0f + client.stateTimeoutSeconds + 0.1f);
            Require(!client.DebugHasLatestState() && !leftArm.DebugJointActive(0) && !leftHand.DebugPalmActive() && !overlay.DebugLeftWristActive(),
                "stale render state did not hide renderers or clear latest state");
            Require(!hud.DebugHasState() && hud.DebugLine(0).Contains("stale"), "status HUD did not show stale state");
            client.ApplyState(sample);
            client.ApplyState(new RenderState {
                v = sample.v + 1,
                stamp = sample.stamp,
                arms = sample.arms,
                hand_render = sample.hand_render,
                op = sample.op,
                status = sample.status
            });
            Require(!client.DebugHasLatestState() && !leftArm.DebugJointActive(0) && !leftHand.DebugPalmActive() && !overlay.DebugLeftWristActive(),
                "schema version mismatch did not hide renderers or clear latest state");
            Require(!hud.DebugHasState() && hud.DebugLine(0).Contains("schema mismatch"), "status HUD did not show schema mismatch");
            client.ApplyState(sample);
            client.ApplyState(new RenderState {v = sample.v});
            Require(!client.DebugHasLatestState() && !leftArm.DebugJointActive(0) && !leftHand.DebugPalmActive() && !overlay.DebugLeftWristActive(),
                "client partial state did not hide renderers or clear latest state");

            client.ApplyState(sample);
            client.ApplyState(new RenderState {
                v = sample.v,
                stamp = sample.stamp,
                arms = new RenderArms {
                    left = new RenderArmState {
                        q = sample.arms.left.q,
                        link_pos = new float[] {0.0f, 0.0f, 0.0f},
                        ee_pos = sample.arms.left.ee_pos,
                        ee_quat = sample.arms.left.ee_quat,
                        cmd_pos = sample.arms.left.cmd_pos,
                        cmd_quat = sample.arms.left.cmd_quat,
                        margins = sample.arms.left.margins
                    },
                    right = sample.arms.right
                },
                hand_render = sample.hand_render,
                op = sample.op,
                status = sample.status
            });
            Require(!client.DebugHasLatestState() && !leftArm.DebugJointActive(0) && !leftHand.DebugPalmActive() && !overlay.DebugLeftWristActive(),
                "client malformed arm state did not hide renderers or clear latest state");

            client.ApplyState(sample);
            client.ApplyState(new RenderState {
                v = sample.v,
                stamp = sample.stamp,
                arms = sample.arms,
                hand_render = new RenderHands {
                    left = new RenderHandState {names = new string[] {"thumb_cmc"}, q = new float[] {1.0f}},
                    right = sample.hand_render.right
                },
                op = sample.op,
                status = sample.status
            });
            Require(!client.DebugHasLatestState() && !leftArm.DebugJointActive(0) && !leftHand.DebugPalmActive() && !overlay.DebugLeftWristActive(),
                "client malformed hand_render state did not hide renderers or clear latest state");

            client.ApplyState(sample);
            client.ApplyState(new RenderState {
                v = sample.v,
                stamp = sample.stamp,
                arms = sample.arms,
                hand_render = sample.hand_render,
                op = new RenderOperatorState {
                    torso_from_head = sample.op.torso_from_head,
                    head_pos = sample.op.head_pos,
                    torso_pos = sample.op.torso_pos,
                    hands = new RenderOperatorHands {
                        left = new RenderOperatorHand {tracked = true, wrist_body = new float[] {0.1f, 0.2f}, raw_wrist = sample.op.hands.left.raw_wrist},
                        right = sample.op.hands.right
                    }
                },
                status = sample.status
            });
            Require(!client.DebugHasLatestState() && !leftArm.DebugJointActive(0) && !leftHand.DebugPalmActive() && !overlay.DebugLeftWristActive(),
                "client malformed operator state did not hide renderers or clear latest state");

            client.ApplyState(sample);
            client.ApplyState(new RenderState {
                v = sample.v,
                stamp = sample.stamp,
                arms = new RenderArms {
                    left = new RenderArmState {
                        q = sample.arms.left.q,
                        link_pos = sample.arms.left.link_pos,
                        ee_pos = new float[] {float.NaN, 0.0f, 0.0f},
                        ee_quat = sample.arms.left.ee_quat,
                        cmd_pos = sample.arms.left.cmd_pos,
                        cmd_quat = sample.arms.left.cmd_quat,
                        margins = sample.arms.left.margins
                    },
                    right = sample.arms.right
                },
                hand_render = sample.hand_render,
                op = sample.op,
                status = sample.status
            });
            Require(!client.DebugHasLatestState() && !leftArm.DebugJointActive(0) && !leftHand.DebugPalmActive() && !overlay.DebugLeftWristActive(),
                "client non-finite arm state did not hide renderers or clear latest state");

            client.ApplyState(sample);
            client.ApplyState(new RenderState {
                v = sample.v,
                stamp = sample.stamp,
                arms = sample.arms,
                hand_render = sample.hand_render,
                op = new RenderOperatorState {
                    torso_from_head = sample.op.torso_from_head,
                    head_pos = sample.op.head_pos,
                    torso_pos = sample.op.torso_pos,
                    hands = new RenderOperatorHands {
                        left = new RenderOperatorHand {tracked = true, wrist_body = new float[] {float.PositiveInfinity, 0.2f, 0.3f}, raw_wrist = sample.op.hands.left.raw_wrist},
                        right = sample.op.hands.right
                    }
                },
                status = sample.status
            });
            Require(!client.DebugHasLatestState() && !leftArm.DebugJointActive(0) && !leftHand.DebugPalmActive() && !overlay.DebugLeftWristActive(),
                "client non-finite operator state did not hide renderers or clear latest state");

            client.ApplyState(sample);
            client.ApplyState(new RenderState {
                v = sample.v,
                stamp = sample.stamp,
                arms = sample.arms,
                hand_render = sample.hand_render,
                op = new RenderOperatorState {
                    torso_from_head = new float[] {float.NaN, -0.35f, 0.0f},
                    head_pos = sample.op.head_pos,
                    torso_pos = sample.op.torso_pos,
                    hands = sample.op.hands
                },
                status = sample.status
            });
            Require(!client.DebugHasLatestState() && !leftArm.DebugJointActive(0) && !leftHand.DebugPalmActive() && !overlay.DebugLeftWristActive(),
                "client non-finite torso_from_head did not hide renderers or clear latest state");

            client.ApplyState(sample);
            client.ApplyState(new RenderState {
                v = sample.v,
                stamp = sample.stamp,
                arms = sample.arms,
                hand_render = sample.hand_render,
                op = sample.op,
                status = new RenderStatus {
                    engaged = sample.status.engaged,
                    tracked = sample.status.tracked,
                    calib = sample.status.calib,
                    hz = float.NaN
                }
            });
            Require(!client.DebugHasLatestState() && !leftArm.DebugJointActive(0) && !leftHand.DebugPalmActive() && !overlay.DebugLeftWristActive(),
                "client non-finite status hz did not hide renderers or clear latest state");

            client.ApplyState(sample);
            client.ApplyState(new RenderState {
                v = sample.v,
                stamp = sample.stamp,
                arms = sample.arms,
                hand_render = sample.hand_render,
                op = sample.op,
                status = new RenderStatus {
                    engaged = sample.status.engaged,
                    tracked = sample.status.tracked,
                    calib = new CalibrationStatus {
                        active = sample.status.calib.active,
                        phase = sample.status.calib.phase,
                        progress = float.PositiveInfinity,
                        remaining = sample.status.calib.remaining,
                        left = sample.status.calib.left,
                        right = sample.status.calib.right,
                        msg = sample.status.calib.msg
                    },
                    hz = sample.status.hz
                }
            });
            Require(!client.DebugHasLatestState() && !leftArm.DebugJointActive(0) && !leftHand.DebugPalmActive() && !overlay.DebugLeftWristActive(),
                "client non-finite calibration status did not hide renderers or clear latest state");

            client.ApplyState(sample);
            client.ApplyJsonAt("{not valid json", 11.0f);
            Require(!client.DebugHasLatestState() && !leftArm.DebugJointActive(0) && !leftHand.DebugPalmActive() && !overlay.DebugLeftWristActive(),
                "malformed json did not hide renderers or clear latest state");
            Require(!hud.DebugHasState() && hud.DebugLine(0).Contains("json error"), "status HUD did not show malformed JSON");

            leftArm.Apply(new RenderArmState {link_pos = new float[] {0.0f, 0.0f, 0.0f, 0.1f, 0.1f, 0.1f}, ee_pos = sample.arms.left.ee_pos}, true, true);
            Require(!leftArm.DebugJointActive(0), "malformed arm link_pos did not hide left arm");
            rightArm.Apply(new RenderArmState {link_pos = new float[] {0.0f, 0.0f, 0.0f, 0.1f, 0.1f, 0.1f}, ee_pos = sample.arms.right.ee_pos}, true, true);
            Require(!rightArm.DebugJointActive(0), "malformed arm link_pos did not hide right arm");

            leftArm.Apply(new RenderArmState {link_pos = sample.arms.left.link_pos, ee_pos = new float[] {float.NaN, 0.0f, 0.0f}}, true, true);
            Require(!leftArm.DebugJointActive(0), "non-finite arm ee_pos did not hide left arm");
            rightArm.Apply(new RenderArmState {link_pos = sample.arms.right.link_pos, ee_pos = new float[] {float.PositiveInfinity, 0.0f, 0.0f}}, true, true);
            Require(!rightArm.DebugJointActive(0), "non-finite arm ee_pos did not hide right arm");

            leftArm.Apply(new RenderArmState {link_pos = sample.arms.left.link_pos, ee_pos = sample.arms.left.ee_pos, cmd_pos = null}, true, true);
            Require(leftArm.DebugJointActive(0) && !leftArm.DebugCmdActive() && !leftArm.DebugCmdLineActive(),
                "null cmd_pos did not hide left command marker/line only");
            leftArm.Apply(new RenderArmState {link_pos = sample.arms.left.link_pos, ee_pos = sample.arms.left.ee_pos, cmd_pos = new float[] {float.NaN, 0.0f, 0.0f}}, true, true);
            Require(!leftArm.DebugJointActive(0), "non-finite arm cmd_pos did not hide left arm");

            leftArm.Apply(sample.arms.left, true, true);
            leftHand.Apply(new RenderHandState {names = new string[] {"thumb_cmc"}, q = new float[] {1.0f}}, sample.arms.left, true);
            Require(!leftHand.DebugPalmActive(), "malformed hand_render did not hide left hand");
            rightArm.Apply(sample.arms.right, true, true);
            rightHand.Apply(new RenderHandState {names = new string[] {"thumb_cmc"}, q = new float[] {1.0f}}, sample.arms.right, true);
            Require(!rightHand.DebugPalmActive(), "malformed hand_render did not hide right hand");

            leftArm.Apply(sample.arms.left, true, true);
            leftHand.Apply(new RenderHandState {
                names = sample.hand_render.left.names,
                q = new float[] {float.NaN, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
            }, sample.arms.left, true);
            Require(!leftHand.DebugPalmActive(), "non-finite hand_render did not hide left hand");
            rightArm.Apply(sample.arms.right, true, true);
            rightHand.Apply(new RenderHandState {
                names = sample.hand_render.right.names,
                q = new float[] {float.PositiveInfinity, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
            }, sample.arms.right, true);
            Require(!rightHand.DebugPalmActive(), "non-finite hand_render did not hide right hand");

            overlay.Apply(new RenderOperatorState {
                torso_from_head = sample.op.torso_from_head,
                head_pos = sample.op.head_pos,
                torso_pos = sample.op.torso_pos,
                hands = new RenderOperatorHands {
                    left = new RenderOperatorHand {tracked = true, wrist_body = new float[] {float.NaN, 0.2f, 0.3f}, raw_wrist = sample.op.hands.left.raw_wrist},
                    right = sample.op.hands.right
                }
            });
            Require(!overlay.DebugLeftWristActive() && !overlay.DebugLeftLineActive(), "operator overlay non-finite wrist_body did not hide left");
        }
        finally
        {
            Object.DestroyImmediate(root);
        }
    }

    private static void ValidateSceneAssetAndBuildSettings()
    {
        TeleopSceneAsset.EnsureRendererScene();
        Require(TeleopSceneAsset.BuildSettingsContainRendererScene(), "renderer scene not registered in build settings");
        Require(GameObject.Find("TeleopRenderer") != null, "renderer scene root missing");
        Require(GameObject.FindObjectOfType<TeleopRenderClient>() != null, "renderer scene client missing");
        Require(GameObject.FindObjectOfType<OperatorVectorRenderer>() != null, "renderer scene operator overlay missing");
    }

    private static void DestroyIfPresent(string name)
    {
        GameObject obj = GameObject.Find(name);
        if (obj != null)
        {
            Object.DestroyImmediate(obj);
        }
    }

    private static string SamplePath()
    {
        return Path.Combine(Application.dataPath, "Editor", "render_state_sample.json");
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new System.Exception("TeleopRenderer validation failed: " + message);
        }
    }

    private static bool VectorClose(Vector3 a, Vector3 b)
    {
        return (a - b).sqrMagnitude < 1e-6f;
    }

    private static bool ColorClose(Color a, Color b)
    {
        Vector4 delta = (Vector4)a - (Vector4)b;
        return delta.sqrMagnitude < 1e-6f;
    }

    private static Vector3 VectorFrom(float[] values, int offset)
    {
        return new Vector3(values[offset], values[offset + 1], values[offset + 2]);
    }
}
#endif
