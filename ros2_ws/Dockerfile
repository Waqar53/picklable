# ROS 2 Jazzy (Ubuntu 24.04, Python 3.12) - the same environment as sam3_ros.
# CPU image for the replay demo; on the robot, build FROM your CUDA/TensorRT base instead.
#
#   docker build -t picklabel-ros ros2_ws
#   docker run --rm -it -v $PWD/runs:/data picklabel-ros \
#     ros2 launch picklabel_ros replay_demo.launch.py source:=/data/frames weights_path:=/data weight_file:=best.pt
FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3-pip python3-opencv \
        ros-jazzy-vision-msgs ros-jazzy-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

# cv_bridge on Jazzy is built against NumPy 1.x (sam3_ros pins the same).
RUN pip install --break-system-packages --no-cache-dir \
        "numpy>=1.26,<2" "ultralytics>=8.3.237" \
        --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /ws
# Same mask message as sam3_ros, so object_masks is wire-compatible.
RUN git clone --depth 1 -b jazzy-devel https://github.com/TeamSOBITS/sobits_interfaces.git src/sobits_interfaces
COPY src src
RUN . /opt/ros/jazzy/setup.sh && colcon build --symlink-install

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
