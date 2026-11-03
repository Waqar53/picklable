"""Record pick before/after frames on the robot, in PickLabel's folder layout.

This is how the data engine attaches to a live cell. The pick executor publishes two JSON
events on ``event_topic`` (std_msgs/String, so no new message package is needed):

    {"event": "grasp_planned", "pick_id": "a17", "grasp_px": [u, v], "class_name": "plastic"}
        -> the latest camera frame (before the arm enters the view) is kept as BEFORE
    {"event": "pick_done", "pick_id": "a17", "grasp_success": true, "belt_shift_px": [dx, dy]}
        -> after ``after_delay_s`` (arm out of view), the latest frame is saved as AFTER

``grasp_success`` comes from gripper feedback (an impedance-controlled gripper that reaches
its closed target with no resistance has missed). ``belt_shift_px`` is the belt motion
between the two frames from the encoder. If it is missing and ``belt_velocity_px_s`` is set,
the shift is velocity x elapsed time.

Each pick becomes ``<out_dir>/<session>/<pick_id>/{before,after}.jpg + pick.json``, ready for
``picklabel autolabel <out_dir> --mode belt``.
"""

import json
import os
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


class PickLogger(Node):
    def __init__(self) -> None:
        super().__init__("picklabel_pick_logger")
        self.declare_parameter("image_topic_name", "image_raw")
        self.declare_parameter("event_topic", "picklabel/pick_events")
        self.declare_parameter("out_dir", os.path.expanduser("~/picklabel_pairs"))
        self.declare_parameter("session", time.strftime("%Y%m%d-%H%M%S"))
        self.declare_parameter("after_delay_s", 0.4)
        self.declare_parameter("belt_velocity_px_s", [0.0, 0.0])
        self.declare_parameter("jpeg_quality", 95)
        p = self.get_parameter
        self.out_dir = os.path.join(p("out_dir").value, p("session").value)
        self.session = p("session").value
        self.after_delay = float(p("after_delay_s").value)
        self.belt_v = [float(v) for v in p("belt_velocity_px_s").value]
        self.quality = int(p("jpeg_quality").value)
        self.bridge = CvBridge()
        self.latest = None  # (msg, receive time)
        self.pending = {}  # pick_id -> (before image, grasp event, time)
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, p("image_topic_name").value, self._on_image, qos)
        self.create_subscription(String, p("event_topic").value, self._on_event, 10)
        self.get_logger().info(f"logging picks to {self.out_dir}")

    def _on_image(self, msg: Image) -> None:
        self.latest = (msg, time.monotonic())

    def _frame(self):
        if self.latest is None:
            return None, None
        msg, t = self.latest
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"), t

    def _on_event(self, msg: String) -> None:
        try:
            ev = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"ignoring non-JSON pick event: {msg.data[:80]}")
            return
        pick_id = str(ev.get("pick_id", int(time.time() * 1000)))
        if ev.get("event") == "grasp_planned":
            img, t = self._frame()
            if img is None:
                self.get_logger().warn("no camera frame yet; pick not logged")
                return
            self.pending[pick_id] = (img, ev, t)
        elif ev.get("event") == "pick_done" and pick_id in self.pending:
            timer = None

            def save() -> None:
                self.destroy_timer(timer)
                self._save(pick_id, ev)

            timer = self.create_timer(self.after_delay, save)

    def _save(self, pick_id: str, done: dict) -> None:
        before, planned, t_before = self.pending.pop(pick_id)
        after, t_after = self._frame()
        if after is None:
            return
        shift = done.get("belt_shift_px")
        if shift is None and any(self.belt_v):
            dt = t_after - t_before
            shift = [self.belt_v[0] * dt, self.belt_v[1] * dt]
        folder = os.path.join(self.out_dir, pick_id)
        os.makedirs(folder, exist_ok=True)
        params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        cv2.imwrite(os.path.join(folder, "before.jpg"), before, params)
        cv2.imwrite(os.path.join(folder, "after.jpg"), after, params)
        meta = {
            "pick_id": pick_id, "session": self.session, "grasp_px": planned.get("grasp_px"),
            "class_name": planned.get("class_name", "plastic"), "grasp_success": done.get("grasp_success"),
            "belt_shift_px": shift, "timestamp": time.time(), "source": "robot",
            "robot_id": done.get("robot_id", planned.get("robot_id")),
        }
        with open(os.path.join(folder, "pick.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self.get_logger().info(f"saved pick {pick_id} (success={meta['grasp_success']}, shift={shift})")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PickLogger()
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
