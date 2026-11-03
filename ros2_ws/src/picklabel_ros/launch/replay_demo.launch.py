"""Demo without a camera: replay recorded frames and run the student on them.

    ros2 launch picklabel_ros replay_demo.launch.py source:=/data/frames weight_file:=best.pt device:=cpu
    ros2 topic hz /picklabel_node/object_boxes
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("picklabel_ros")
    args = [
        DeclareLaunchArgument("source", description="folder of images or a video file"),
        DeclareLaunchArgument("rate_hz", default_value="15.0"),
        DeclareLaunchArgument("weight_file", default_value="best.pt"),
        DeclareLaunchArgument("weights_path", default_value=os.path.join(share, "weights")),
        DeclareLaunchArgument("device", default_value="cpu"),
        DeclareLaunchArgument("threshold", default_value="0.5"),
        DeclareLaunchArgument("imgsz", default_value="640"),
    ]
    replay = Node(
        package="picklabel_ros", executable="replay_node", name="picklabel_replay", output="screen",
        parameters=[{"source": LaunchConfiguration("source"), "rate_hz": LaunchConfiguration("rate_hz"),
                     "image_topic_name": "camera/color/image_raw"}],
    )
    detector = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, "launch", "picklabel.launch.py")),
        launch_arguments={
            "weight_file": LaunchConfiguration("weight_file"),
            "weights_path": LaunchConfiguration("weights_path"),
            "device": LaunchConfiguration("device"),
            "threshold": LaunchConfiguration("threshold"),
            "imgsz": LaunchConfiguration("imgsz"),
            "half": "false",
            "image_reliability": "reliable",
            "publish_mask": "true",
        }.items(),
    )
    return LaunchDescription(args + [replay, detector])
