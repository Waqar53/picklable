"""Publish a folder of images or a video file as a camera topic, looping.

There is no ZED or arm here, so the demo replays recorded frames:

    ros2 run picklabel_ros replay_node --ros-args -p source:=/data/tray_video.mp4 -p rate_hz:=15.0
"""

import os

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class Replay(Node):
    def __init__(self) -> None:
        super().__init__("picklabel_replay")
        self.declare_parameter("source", "")
        self.declare_parameter("image_topic_name", "camera/color/image_raw")
        self.declare_parameter("rate_hz", 15.0)
        self.declare_parameter("frame_id", "camera_color_optical_frame")
        self.declare_parameter("loop", True)
        p = self.get_parameter
        self.source = os.path.expanduser(p("source").value)
        self.frame_id = p("frame_id").value
        self.loop = bool(p("loop").value)
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, p("image_topic_name").value, 1)
        self.files, self.cap, self.i = [], None, 0
        if os.path.isdir(self.source):
            for root, _, names in os.walk(self.source):
                self.files += [os.path.join(root, n) for n in names if n.lower().endswith(EXTS)]
            self.files.sort()
        elif self.source:
            self.cap = cv2.VideoCapture(self.source)
        if not self.files and (self.cap is None or not self.cap.isOpened()):
            raise SystemExit(f"nothing to replay at source={self.source!r}")
        self.create_timer(1.0 / float(p("rate_hz").value), self._tick)
        self.get_logger().info(f"replaying {self.source} ({len(self.files) or 'video'}) on {self.pub.topic_name}")

    def _next(self):
        if self.files:
            if self.i >= len(self.files):
                if not self.loop:
                    return None
                self.i = 0
            img = cv2.imread(self.files[self.i])
            self.i += 1
            return img
        ok, img = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, img = self.cap.read()
        return img if ok else None

    def _tick(self) -> None:
        img = self._next()
        if img is None:
            return
        msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Replay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
