"""PickLabel student detector, launched with the same arguments as sam3_ros' sam3.launch.py.

    # side by side with SAM 3
    ros2 launch picklabel_ros picklabel.launch.py weight_file:=best.pt image_topic_name:=/zed/zed_node/rgb/image_rect_color
    # literal drop-in: same node name, so /sam3_node/object_boxes etc. come from the student
    ros2 launch picklabel_ros picklabel.launch.py node_name:=sam3_node publish_mask:=true use_mask_to_3d:=true
"""

import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import AndSubstitution, LaunchConfiguration
from launch_ros.actions import Node

ARGS = [
    ("namespace", "", "Namespace for the nodes"),
    ("node_name", "picklabel_node", "Node name; set to sam3_node for a literal drop-in replacement"),
    ("use_sim_time", "false", "Use simulation clock"),
    ("image_topic_name", "camera/color/image_raw", "sensor_msgs/Image input (ZED: zed/zed_node/rgb/image_rect_color)"),
    ("weight_file", "best.pt", "Student weights: .pt, .onnx or TensorRT .engine"),
    ("weights_path", "", "Directory holding weight_file (default: this package's share/weights)"),
    ("auto_configure_2d", "true", "Configure the detector lifecycle node on startup"),
    ("auto_activate_2d", "true", "Activate the detector lifecycle node on startup"),
    ("auto_configure_3d", "false", "Configure image_to_position on startup"),
    ("auto_activate_3d", "false", "Activate image_to_position on startup"),
    ("image_show", "false", "Show an OpenCV window with detections"),
    ("threshold", "0.5", "Score threshold; use the calibrated value from picklabel eval-student"),
    ("class_thresholds", "['']", "Per-class thresholds, e.g. \"['plastic=0.45']\""),
    ("prompt_text", "['']", "Class filter (student class names); empty = all classes"),
    ("inference_hz", "15.0", "Inference rate (timer-driven, latest frame)"),
    ("imgsz", "1024", "Student input size"),
    ("half", "true", "FP16 (CUDA only)"),
    ("publish_mask", "false", "Publish object_masks (sobits_interfaces/DetectMaskArray)"),
    ("publish_mask_pixels", "false", "Include pixel_x / pixel_y in each mask"),
    ("publish_mask_image", "false", "Include the mono8 mask image in each mask"),
    ("image_reliability", "best_effort", "QoS reliability of the image subscription"),
    ("device", "cuda", "cuda | cuda:0 | cpu"),
    ("use_bbox_to_3d", "false", "Launch image_to_position bbox_to_3d (if installed)"),
    ("use_mask_to_3d", "false", "Launch image_to_position mask_to_3d (needs publish_mask)"),
]


def generate_launch_description():
    share = get_package_share_directory("picklabel_ros")
    launch_args = [
        DeclareLaunchArgument(name, default_value=(os.path.join(share, "weights") if name == "weights_path"
                                                   else default), description=desc)
        for name, default, desc in ARGS
    ]
    cfg = {name: LaunchConfiguration(name) for name, _, _ in ARGS}

    detector = Node(
        package="picklabel_ros",
        executable="detector_node",
        name=cfg["node_name"],
        namespace=cfg["namespace"],
        parameters=[{
            "weight_file": cfg["weight_file"],
            "weights_path": cfg["weights_path"],
            "auto_configure": cfg["auto_configure_2d"],
            "auto_activate": cfg["auto_activate_2d"],
            "image_topic_name": cfg["image_topic_name"],
            "threshold": cfg["threshold"],
            "class_thresholds": cfg["class_thresholds"],
            "half": cfg["half"],
            "image_show": cfg["image_show"],
            "prompt_text": cfg["prompt_text"],
            "inference_hz": cfg["inference_hz"],
            "imgsz": cfg["imgsz"],
            "publish_mask": cfg["publish_mask"],
            "publish_mask_pixels": cfg["publish_mask_pixels"],
            "publish_mask_image": cfg["publish_mask_image"],
            "image_reliability": cfg["image_reliability"],
            "device": cfg["device"],
            "use_sim_time": cfg["use_sim_time"],
        }],
        output="screen",
    )

    actions = launch_args + [detector]
    try:  # image_to_position (TeamSOBITS) turns 2D detections into 3D positions, as in sam3.launch.py
        i2p = get_package_share_directory("image_to_position")
        for kind, condition in (("bbox_to_3d", IfCondition(cfg["use_bbox_to_3d"])),
                                ("mask_to_3d", IfCondition(AndSubstitution(cfg["publish_mask"],
                                                                           cfg["use_mask_to_3d"])))):
            actions.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(i2p, "launch", f"{kind}.launch.py")),
                launch_arguments={"namespace": cfg["namespace"],
                                  "params_file": os.path.join(i2p, "config", f"{kind}.yaml"),
                                  "auto_configure_3d": cfg["auto_configure_3d"],
                                  "auto_activate_3d": cfg["auto_activate_3d"]}.items(),
                condition=condition,
            ))
    except PackageNotFoundError:
        pass
    return LaunchDescription(actions)
