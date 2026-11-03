"""Drive the real detector node through configure -> activate -> frame -> publish, with fake ROS modules.

No ROS install needed: rclpy, the message packages and cv_bridge are replaced by minimal fakes, so
the node's own logic (parameter handling, lifecycle, message filling) runs exactly as written.
"""

import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest


class Auto(SimpleNamespace):
    """Message stand-in: unknown attributes spring into existence as nested messages."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        child = Auto()
        setattr(self, name, child)
        return child


def msg_class(**list_fields):
    class Msg(Auto):
        def __init__(self, **kw):
            super().__init__(**{k: [] for k in list_fields}, **kw)

    return Msg


class FakePublisher:
    def __init__(self, topic):
        self.topic, self.sent = topic, []

    def publish(self, msg):
        self.sent.append(msg)


class FakeLifecycleNode:
    def __init__(self, name):
        self._name, self._params, self.publishers = name, {}, {}
        self._state_machine = SimpleNamespace(current_state=(0, "unconfigured"))
        self.logs = []

    def declare_parameter(self, name, value):
        self._params.setdefault(name, value)

    def get_parameter(self, name):
        return SimpleNamespace(value=self._params[name])

    def get_name(self):
        return self._name

    def get_logger(self):
        log = lambda *a, **k: self.logs.append(a[0])  # noqa: E731
        return SimpleNamespace(info=log, warn=log, error=log, debug=log)

    def get_clock(self):
        class T:
            def __init__(self, ns):
                self.ns = ns

            def __sub__(self, other):
                return SimpleNamespace(nanoseconds=self.ns - other.ns)

        return SimpleNamespace(now=lambda: T(0))

    def create_lifecycle_publisher(self, typ, topic, depth):
        self.publishers[topic] = FakePublisher(topic)
        return self.publishers[topic]

    def destroy_lifecycle_publisher(self, pub):
        pass

    def create_subscription(self, typ, topic, cb, qos):
        self.subscribed = topic
        return object()

    def create_timer(self, period, cb):
        self.timer_period = period
        return object()

    def destroy_subscription(self, s):
        pass

    def destroy_timer(self, t):
        pass

    def add_on_set_parameters_callback(self, cb):
        return cb

    def remove_on_set_parameters_callback(self, cb):
        pass

    def on_configure(self, state):
        return "SUCCESS"

    def on_activate(self, state):
        return "SUCCESS"

    def on_deactivate(self, state):
        return "SUCCESS"

    def on_cleanup(self, state):
        return "SUCCESS"

    def on_shutdown(self, state):
        return "SUCCESS"


@pytest.fixture
def node_module(monkeypatch):
    mods = {
        "rclpy": types.ModuleType("rclpy"),
        "rclpy.lifecycle": types.ModuleType("rclpy.lifecycle"),
        "rclpy.qos": types.ModuleType("rclpy.qos"),
        "rcl_interfaces": types.ModuleType("rcl_interfaces"),
        "rcl_interfaces.msg": types.ModuleType("rcl_interfaces.msg"),
        "sensor_msgs": types.ModuleType("sensor_msgs"),
        "sensor_msgs.msg": types.ModuleType("sensor_msgs.msg"),
        "vision_msgs": types.ModuleType("vision_msgs"),
        "vision_msgs.msg": types.ModuleType("vision_msgs.msg"),
        "geometry_msgs": types.ModuleType("geometry_msgs"),
        "geometry_msgs.msg": types.ModuleType("geometry_msgs.msg"),
        "sobits_interfaces": types.ModuleType("sobits_interfaces"),
        "sobits_interfaces.msg": types.ModuleType("sobits_interfaces.msg"),
        "cv_bridge": types.ModuleType("cv_bridge"),
    }
    mods["rclpy.lifecycle"].LifecycleNode = FakeLifecycleNode
    mods["rclpy.lifecycle"].LifecycleState = object
    mods["rclpy.lifecycle"].TransitionCallbackReturn = SimpleNamespace(SUCCESS="SUCCESS", FAILURE="FAILURE",
                                                                       ERROR="ERROR")
    q = mods["rclpy.qos"]
    q.QoSProfile = lambda **kw: SimpleNamespace(**kw)
    q.QoSReliabilityPolicy = SimpleNamespace(BEST_EFFORT=1, RELIABLE=2, SYSTEM_DEFAULT=0, BEST_AVAILABLE=3, UNKNOWN=4)
    q.QoSHistoryPolicy = SimpleNamespace(KEEP_LAST=1)
    q.QoSDurabilityPolicy = SimpleNamespace(VOLATILE=2)
    mods["rcl_interfaces.msg"].SetParametersResult = lambda **kw: SimpleNamespace(**kw)
    mods["sensor_msgs.msg"].Image = msg_class()
    mods["vision_msgs.msg"].Detection2DArray = msg_class(detections=True)
    mods["vision_msgs.msg"].Detection2D = msg_class(results=True)
    mods["vision_msgs.msg"].ObjectHypothesisWithPose = msg_class()
    mods["geometry_msgs.msg"].Point = lambda **kw: SimpleNamespace(**kw)
    mods["geometry_msgs.msg"].Quaternion = lambda **kw: SimpleNamespace(**kw)
    mods["sobits_interfaces.msg"].DetectMask = msg_class(results=True, pixel_x=True, pixel_y=True)
    mods["sobits_interfaces.msg"].DetectMaskArray = msg_class(masks=True)

    class Bridge:
        def imgmsg_to_cv2(self, msg, desired_encoding="bgr8"):
            return msg.data

        def cv2_to_imgmsg(self, img, encoding="bgr8"):
            return Auto(data=img, encoding=encoding)

    mods["cv_bridge"].CvBridge = Bridge
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "picklabel_ros.detector_node", raising=False)
    return importlib.import_module("picklabel_ros.detector_node")


def test_node_publishes_sam3_ros_compatible_messages(node_module, tmp_path, monkeypatch):
    from picklabel_ros.inference import Detection

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    mask = np.zeros((60, 80), bool)
    mask[10:30, 20:50] = True

    class FakeDetector:
        def __init__(self, *a, **k):
            self.model = SimpleNamespace(names={0: "plastic"})

        def __call__(self, image, threshold, class_filter=None, class_thresholds=None):
            return [Detection("plastic", 0.83, mask, (20, 10, 30, 20), "plastic_0")]

    monkeypatch.setattr(node_module, "StudentDetector", FakeDetector)
    node = node_module.PickLabelDetector()
    node._params.update(weights_path=str(tmp_path), weight_file="best.pt", image_topic_name="/zed/rgb",
                        publish_mask=True)
    assert node.on_configure(None) == "SUCCESS"
    assert set(node.publishers) == {"picklabel_node/object_boxes", "picklabel_node/object_masks",
                                    "picklabel_node/detected_image"}
    assert node.on_activate(None) == "SUCCESS" and node.subscribed == "/zed/rgb"
    assert abs(node.timer_period - 1 / 15.0) < 1e-9

    frame = Auto(data=np.zeros((60, 80, 3), np.uint8), header=Auto(stamp=1))
    node._on_image(frame)
    node._on_timer()
    node._on_timer()  # same stamp: not processed twice

    boxes = node.publishers["picklabel_node/object_boxes"].sent
    assert len(boxes) == 1
    det = boxes[0].detections[0]
    assert det.id == "plastic_0"
    assert (det.bbox.center.position.x, det.bbox.center.position.y) == (35.0, 20.0)
    assert (det.bbox.size_x, det.bbox.size_y) == (30.0, 20.0)
    hyp = det.results[0]
    assert hyp.hypothesis.class_id == "plastic" and hyp.hypothesis.score == pytest.approx(0.83)
    assert hyp.pose.pose.position.z == -1.0 and hyp.pose.pose.orientation.w == 1.0

    masks = node.publishers["picklabel_node/object_masks"].sent[0].masks
    assert masks[0].instance_id == "plastic_0" and len(masks[0].pixel_x) == mask.sum()
    assert masks[0].mask.encoding == "mono8" and masks[0].mask.data.max() == 255
    assert len(node.publishers["picklabel_node/detected_image"].sent) == 1


def test_node_refuses_missing_weights(node_module, tmp_path):
    node = node_module.PickLabelDetector()
    node._params.update(weights_path=str(tmp_path), weight_file="missing.pt")
    assert node.on_configure(None) == "FAILURE"


def test_runtime_threshold_update_is_validated(node_module, tmp_path):
    node = node_module.PickLabelDetector()
    node.threshold = 0.5
    bad = node.on_parameter_update([SimpleNamespace(name="threshold", value=1.5)])
    good = node.on_parameter_update([SimpleNamespace(name="threshold", value=0.42)])
    assert bad.successful is False and good.successful is True and node.threshold == 0.42
