# picklabel_ros (ROS 2 Jazzy)

Three nodes:

| Node | What it does |
|---|---|
| `detector_node` | The YOLO-seg student as a **drop-in for `sam3_ros`**: lifecycle node, the same parameters, the same topics and message types, real confidence scores |
| `pick_logger_node` | Records each pick's before/after frames + grasp pixel + gripper feedback + belt shift, in PickLabel's folder layout |
| `replay_node` | Publishes a folder of frames or a video as a camera topic, for demos without a camera |

## Interface (identical to sam3_ros)

| Publication | Type |
|---|---|
| `<node>/object_boxes` | `vision_msgs/Detection2DArray`: `id = "<class>_<i>"`, bbox = mask bounding rect, `results[0].hypothesis.{class_id, score}`, pose `(cx, cy, -1)` |
| `<node>/object_masks` | `sobits_interfaces/DetectMaskArray` (when `publish_mask:=true`; needs `sobits_interfaces` in the workspace) |
| `<node>/detected_image` | `sensor_msgs/Image` |

Parameters shared with `sam3_ros`: `weight_file`, `weights_path`, `auto_configure`, `auto_activate`,
`threshold`, `half`, `image_topic_name`, `prompt_text` (here a class filter, empty = all),
`image_show`, `inference_hz`, `publish_mask`, `publish_mask_pixels`, `publish_mask_image`,
`image_reliability`, `device`. Extra: `imgsz`, `max_det`, `class_thresholds` (`"['plastic=0.45']"`).

Runtime changes work the same way:

```bash
ros2 param set /sam3_node threshold 0.42
ros2 param set /sam3_node class_thresholds "['plastic=0.45']"
```

## Swap it in

```bash
# side by side, to compare against SAM 3 on the same image stream
ros2 launch picklabel_ros picklabel.launch.py image_topic_name:=/zed/zed_node/rgb/image_rect_color weights_path:=/weights weight_file:=best.engine

# literal drop-in: same node name, so /sam3_node/object_boxes now comes from the student
ros2 launch picklabel_ros picklabel.launch.py node_name:=sam3_node publish_mask:=true use_mask_to_3d:=true
```

`weight_file` may be `.pt`, `.onnx` or a TensorRT `.engine` (`yolo export format=engine half=True`
on the robot's GPU).

## Log picks on the robot

The pick executor publishes two JSON strings on `picklabel/pick_events`:

```json
{"event": "grasp_planned", "pick_id": "a17", "grasp_px": [812, 455], "class_name": "plastic"}
{"event": "pick_done", "pick_id": "a17", "grasp_success": true, "belt_shift_px": [214.0, 1.5]}
```

```bash
ros2 run picklabel_ros pick_logger_node --ros-args -p image_topic_name:=/zed/zed_node/rgb/image_rect_color \
    -p out_dir:=/data/picks -p after_delay_s:=0.4
picklabel autolabel /data/picks --out runs/autolabel/robot --mode belt --sam2 sam2.1_t.pt --push
```

## Demo without a camera (Docker)

```bash
docker build -t picklabel-ros ros2_ws
docker run --rm -it -v $PWD/runs:/data picklabel-ros \
  ros2 launch picklabel_ros replay_demo.launch.py source:=/data/frames weights_path:=/data weight_file:=best.pt
# second shell
docker exec -it <container> bash -c "source /ws/install/setup.bash && ros2 topic hz /picklabel_node/object_boxes"
```

## Status

The ROS-free core (`inference.py`: result conversion, thresholds, class filter, `prompt_text`
parsing identical to sam3_ros) is unit-tested in `tests/test_ros_and_sam3.py`. The detector node
itself runs in `tests/test_ros_node_fake.py` against fake `rclpy` / message modules: configure,
activate, one frame, and every published field checked against the sam3_ros conventions. It has
not run on a real ROS 2 install, a ZED or Grip's arm.
