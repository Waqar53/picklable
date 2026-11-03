# How PickLabel maps onto Grip's public stack

Everything here comes from Grip Robotics' public GitHub forks, read in November 2026. Commit
references are facts. Anything about the private pipeline is an inference and is marked as one.

## What the forks show

| Fork | Upstream | Grip's own commits (fact) | What it suggests (inference) |
|---|---|---|---|
| `sam3_ros` | TeamSOBITS/sam3_ros | `18db77a` "Publish SAM3 model confidence scores": `hypothesis.score` carries the real SAM 3 confidence instead of `1.0`, with checks for missing / non-finite scores | Perception output is consumed by score, so thresholds and calibration matter downstream |
| `zed-ros2-wrapper` | stereolabs | `c636555` sets `auto_exposure_gain: false` (fixed exposure 80, gain 80); Docker + TensorRT runtime fixes; QoS overrides for raw image publishers | A fixed-exposure ZED, containerised. Auto white balance is still on |
| `frcobot_ros2` | FAIR-INNOVATION | ServoJ UDP streaming, ServoJ tracking, bus-gripper feedback on `RobotNonrtState`, TCP/finger frames, multi-robot controller support, Fairino 10 / 16 MoveIt configs | Fairino arms tracking a moving belt, more than one arm per line, gripper state available |
| `curobo` | NVlabs | `2a8baee` "Add nonzero boundary velocities to cspace planning" | Trajectories that start/end in motion: picking from a belt without stopping |
| `candle_ros2` | mabrobotics | gripper open/close + init services, `ConfigureGripper` (impedance kp/kd, velocity / torque limits), `SetGripperTargets`, batched joint-state reads | A custom gripper on MAB md80 drives under impedance control |

`sam3_ros` defaults: threshold 0.75, 5 Hz in the node (10 Hz in the launch file), one SAM 3
forward pass per text prompt, every prompt's masks published independently.

## Design decisions this drove

1. **Drop-in ROS 2 node, not a new interface.** `ros2_ws/src/picklabel_ros` mirrors `sam3_ros`:
   lifecycle node, the same parameter names, the same three topics and message types
   (`Detection2DArray`, `sobits_interfaces/DetectMaskArray`, `Image`), the same
   `instance_id` / bbox / pose conventions, and real confidence scores. Launched with
   `node_name:=sam3_node`, downstream consumers (`image_to_position`, pick logic) don't change.
2. **Confidence matters, so it is measured.** The SAM 3 baseline sweeps the threshold and
   reports recall at precision >= 0.90 against the node's default 0.75. The tuned value can be
   set live with `ros2 param set /sam3_node threshold <t>`.
3. **Cross-prompt duplicates are counted.** `sam3_ros` publishes each prompt's masks
   separately, so one bag matched by "plastic bag" and by "plastic film" is two detections,
   two candidate picks. `picklabel sam3-baseline` reports those duplicates (`node` mode) and the
   effect of cross-prompt mask NMS (`nms` mode).
4. **Fixed exposure helps; white balance still drifts.** Before/after differencing relies on
   stable imaging, which the `auto_exposure_gain: false` commit provides. Auto white balance is
   still on, so the change map applies per-channel gains first. On synthetic trays, colour drift
   alone pushed false change to 7-12% of the frame without it, and 0% with it.
5. **Belt mode uses the robot's own tracking.** The ServoJ tracking and nonzero-boundary-velocity
   commits imply the arm already knows belt motion. `align.mode: belt` takes the encoder
   displacement as a prior and refines it with phase correlation; `pick_logger_node` records it
   per pick.
6. **Gripper feedback is a label signal.** An impedance gripper that reaches its closed target
   without resistance has missed. PickLabel turns "gripper empty + no visible change" into a
   hard example (likely a perception false positive) and never auto-accepts "gripper empty but
   the scene changed" (the item was pushed, not picked).
7. **ZED-native option.** The ZED wrapper can run a YOLO-like ONNX
   (`CUSTOM_YOLOLIKE_BOX_OBJECTS`) with the SDK's own 3D tracking. `picklabel export --kind zed`
   writes that ONNX and the exact `custom_object_detection.yaml` block. Boxes only, so it sits
   next to the mask node rather than replacing it.

## What I could not see or test

The real pipeline code is private. I have no ZED, no Fairino arm and no plant data. The ROS 2
node is checked by tests that run its code against fake `rclpy` and message modules, and it is
meant to run on replayed frames (`replay_demo.launch.py`). It has not run on a real ROS install or
Grip hardware.
