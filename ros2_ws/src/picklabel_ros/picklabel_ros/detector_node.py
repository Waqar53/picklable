"""PickLabel student detector: a drop-in replacement for the sam3_ros node.

Same lifecycle (configure -> activate -> deactivate -> cleanup, with auto_configure /
auto_activate), same parameter names, and the same three publications with the same message
types and field conventions:

    <node>/object_boxes    vision_msgs/Detection2DArray        (hypothesis.score = model confidence)
    <node>/object_masks    sobits_interfaces/DetectMaskArray   (when publish_mask is true)
    <node>/detected_image  sensor_msgs/Image

Launched with ``node_name:=sam3_node``, consumers of sam3_ros (image_to_position's
bbox_to_3d / mask_to_3d, the pick logic) see no difference except speed.

Differences, all opt-in: ``prompt_text`` filters the student's classes instead of prompting
(empty = all classes); ``class_thresholds`` sets per-class thresholds (``["plastic=0.45"]``);
``imgsz`` and ``max_det`` control the student.
"""

import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Quaternion
from rcl_interfaces.msg import SetParametersResult
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from picklabel_ros.inference import StudentDetector, draw, parse_class_thresholds, parse_prompt_text

try:
    from sobits_interfaces.msg import DetectMask, DetectMaskArray

    HAVE_MASK_MSGS = True
except ImportError:  # boxes still work; masks need sobits_interfaces in the workspace
    HAVE_MASK_MSGS = False


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class PickLabelDetector(LifecycleNode):
    _RELIABILITY_MAP = {
        "best_effort": QoSReliabilityPolicy.BEST_EFFORT,
        "reliable": QoSReliabilityPolicy.RELIABLE,
        "system_default": QoSReliabilityPolicy.SYSTEM_DEFAULT,
        "best_available": QoSReliabilityPolicy.BEST_AVAILABLE,
        "unknown": QoSReliabilityPolicy.UNKNOWN,
    }

    def __init__(self) -> None:
        super().__init__("picklabel_node")
        self.declare_parameter("weight_file", "best.pt")
        self.declare_parameter("weights_path", "")
        self.declare_parameter("auto_configure", True)
        self.declare_parameter("auto_activate", True)
        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("half", True)
        self.declare_parameter("image_topic_name", "image_raw")
        self.declare_parameter("prompt_text", [""])
        self.declare_parameter("image_show", False)
        self.declare_parameter("inference_hz", 15.0)
        self.declare_parameter("publish_mask", True)
        self.declare_parameter("publish_mask_pixels", True)
        self.declare_parameter("publish_mask_image", True)
        self.declare_parameter("image_reliability", "best_effort")
        self.declare_parameter("device", _default_device())
        self.declare_parameter("imgsz", 1024)
        self.declare_parameter("max_det", 100)
        self.declare_parameter("class_thresholds", [""])

        self._detector = None
        self._sub = None
        self._timer = None
        self._param_cb = None
        self._pub_det = self._pub_mask = self._pub_img = None
        self._latest_msg = None
        self._last_processed_stamp = None
        self._busy = False
        self._ms_ema = None

    # ---------------------------------------------------------------- lifecycle

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        p = self.get_parameter
        self.weight_file = os.path.join(p("weights_path").value, p("weight_file").value)
        self.threshold = float(p("threshold").value)
        self.half = bool(p("half").value)
        self.image_topic = p("image_topic_name").value
        self.class_filter = parse_prompt_text(p("prompt_text").value)
        self.class_thresholds = parse_class_thresholds(p("class_thresholds").value)
        self.image_show = bool(p("image_show").value)
        self.inference_hz = max(float(p("inference_hz").value), 0.1)
        self.publish_mask = bool(p("publish_mask").value)
        self.publish_mask_pixels = bool(p("publish_mask_pixels").value)
        self.publish_mask_image = bool(p("publish_mask_image").value)
        self.image_reliability = p("image_reliability").value
        self.device = p("device").value
        self.imgsz = int(p("imgsz").value)
        self.max_det = int(p("max_det").value)

        if not os.path.exists(self.weight_file):
            self.get_logger().error(f"Model file not found: {self.weight_file}")
            return TransitionCallbackReturn.FAILURE
        if not 0.0 < self.threshold <= 1.0:
            self.get_logger().error(f"threshold must be in (0.0, 1.0], got {self.threshold}")
            return TransitionCallbackReturn.FAILURE
        if self.image_reliability not in self._RELIABILITY_MAP:
            self.get_logger().error(f"image_reliability must be one of {list(self._RELIABILITY_MAP)}")
            return TransitionCallbackReturn.FAILURE
        if self.publish_mask and not HAVE_MASK_MSGS:
            self.get_logger().warn("sobits_interfaces not found: object_masks disabled, boxes still published")
            self.publish_mask = False

        self.get_logger().info(
            f"weights={self.weight_file} threshold={self.threshold} class_thresholds={self.class_thresholds} "
            f"classes={self.class_filter or 'all'} imgsz={self.imgsz} hz={self.inference_hz} device={self.device} "
            f"topic={self.image_topic}")

        self.image_qos = QoSProfile(reliability=self._RELIABILITY_MAP[self.image_reliability],
                                    history=QoSHistoryPolicy.KEEP_LAST, durability=QoSDurabilityPolicy.VOLATILE,
                                    depth=1)
        name = self.get_name()
        self._pub_det = self.create_lifecycle_publisher(Detection2DArray, name + "/object_boxes", 1)
        if HAVE_MASK_MSGS:
            self._pub_mask = self.create_lifecycle_publisher(DetectMaskArray, name + "/object_masks", 1)
        self._pub_img = self.create_lifecycle_publisher(Image, name + "/detected_image", 1)
        self._bridge = CvBridge()
        self._param_cb = self.add_on_set_parameters_callback(self.on_parameter_update)
        return super().on_configure(state)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        try:
            half = self.half and "cuda" in str(self.device)
            self._detector = StudentDetector(self.weight_file, self.device, half, self.imgsz, self.max_det)
            warm = np.zeros((self.imgsz, self.imgsz, 3), np.uint8)
            self._detector(warm, self.threshold)  # first call compiles kernels; keep it out of the loop
            self.get_logger().info(f"Student loaded: {self.weight_file} ({self._detector.model.names})")
        except Exception as e:  # noqa: BLE001 - report any load failure as a lifecycle error
            self._detector = None
            self.get_logger().error(f"Failed to load student: {e}")
            return TransitionCallbackReturn.ERROR
        self._sub = self.create_subscription(Image, self.image_topic, self._on_image, self.image_qos)
        self._timer = self.create_timer(1.0 / self.inference_hz, self._on_timer)
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self._stop_io()
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self._release()
        return super().on_cleanup(state)

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self._stop_io()
        self._release()
        return super().on_shutdown(state)

    def _stop_io(self) -> None:
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        if self._timer is not None:
            self.destroy_timer(self._timer)
            self._timer = None
        self._busy = False

    def _release(self) -> None:
        self._detector = None
        if self._param_cb is not None:
            self.remove_on_set_parameters_callback(self._param_cb)
            self._param_cb = None
        for attr in ("_pub_det", "_pub_mask", "_pub_img"):
            pub = getattr(self, attr)
            if pub is not None:
                self.destroy_lifecycle_publisher(pub)
                setattr(self, attr, None)

    # ---------------------------------------------------------------- runtime parameters

    def on_parameter_update(self, params) -> SetParametersResult:
        for param in params:
            if param.name == "threshold":
                v = float(param.value)
                if not 0.0 < v <= 1.0:
                    return SetParametersResult(successful=False, reason="threshold must be in (0.0, 1.0]")
                self.threshold = v
            elif param.name == "prompt_text":
                self.class_filter = parse_prompt_text(param.value)
            elif param.name == "class_thresholds":
                try:
                    self.class_thresholds = parse_class_thresholds(param.value)
                except ValueError as e:
                    return SetParametersResult(successful=False, reason=str(e))
            elif param.name == "inference_hz":
                v = float(param.value)
                if v <= 0.0:
                    return SetParametersResult(successful=False, reason="inference_hz must be > 0")
                self.inference_hz = max(v, 0.1)
                if self._timer is not None:
                    self.destroy_timer(self._timer)
                    self._timer = self.create_timer(1.0 / self.inference_hz, self._on_timer)
            elif param.name in ("publish_mask", "publish_mask_pixels", "publish_mask_image"):
                setattr(self, param.name, bool(param.value) and (param.name != "publish_mask" or HAVE_MASK_MSGS))
            elif param.name in ("weight_file", "weights_path", "imgsz", "device", "image_reliability"):
                if self._state_machine.current_state[1] != "unconfigured":
                    return SetParametersResult(successful=False,
                                               reason=f"{param.name} can only change when unconfigured")
        return SetParametersResult(successful=True)

    # ---------------------------------------------------------------- processing

    def _on_image(self, msg: Image) -> None:
        self._latest_msg = msg

    def _on_timer(self) -> None:
        msg = self._latest_msg
        if self._detector is None or self._busy:
            return
        if msg is None:
            self.get_logger().warn(f"Waiting for image on '{self.image_topic}'", throttle_duration_sec=10.0)
            return
        if msg.header.stamp == self._last_processed_stamp:
            return
        self._busy = True
        try:
            self._process(msg)
        except Exception as e:  # noqa: BLE001 - keep the node alive on a bad frame
            self.get_logger().error(f"inference failed: {e}")
        finally:
            self._last_processed_stamp = msg.header.stamp
            self._busy = False

    def _process(self, msg: Image) -> None:
        image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        t0 = self.get_clock().now()
        dets = self._detector(image, self.threshold, self.class_filter, self.class_thresholds)
        ms = (self.get_clock().now() - t0).nanoseconds / 1e6
        self._ms_ema = ms if self._ms_ema is None else 0.9 * self._ms_ema + 0.1 * ms
        self.get_logger().debug(f"{len(dets)} detections in {ms:.1f} ms (ema {self._ms_ema:.1f})")

        boxes = Detection2DArray(header=msg.header)
        masks = DetectMaskArray(header=msg.header) if self.publish_mask else None
        for d in dets:
            x, y, w, h = d.bbox_xywh
            det = Detection2D(header=msg.header, id=d.instance_id)
            det.bbox.center.position.x = float(x + w / 2.0)
            det.bbox.center.position.y = float(y + h / 2.0)
            det.bbox.size_x, det.bbox.size_y = float(w), float(h)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = d.class_name
            hyp.hypothesis.score = d.score
            hyp.pose.pose.position = Point(x=det.bbox.center.position.x, y=det.bbox.center.position.y, z=-1.0)
            hyp.pose.pose.orientation = Quaternion(w=1.0)
            det.results.append(hyp)
            boxes.detections.append(det)
            if masks is not None:
                m = DetectMask(instance_id=d.instance_id)
                if self.publish_mask_pixels:
                    ys, xs = np.nonzero(d.mask)
                    m.pixel_x, m.pixel_y = xs.astype(np.int32).tolist(), ys.astype(np.int32).tolist()
                if self.publish_mask_image:
                    m.mask = self._bridge.cv2_to_imgmsg(d.mask.astype(np.uint8) * 255, encoding="mono8")
                    m.mask.header = msg.header
                mh = ObjectHypothesisWithPose()
                mh.hypothesis.class_id = d.class_name
                mh.hypothesis.score = d.score
                m.results.append(mh)
                masks.masks.append(m)
        if masks is not None:
            self._pub_mask.publish(masks)
        self._pub_det.publish(boxes)
        vis = draw(image, dets)
        out = self._bridge.cv2_to_imgmsg(vis, "bgr8")
        out.header = msg.header
        self._pub_img.publish(out)
        if self.image_show:
            cv2.imshow(self.get_name(), vis)
            cv2.waitKey(1)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PickLabelDetector()
    configured = True
    if node.get_parameter("auto_configure").value or node.get_parameter("auto_activate").value:
        configured = node.trigger_configure() == TransitionCallbackReturn.SUCCESS
    if node.get_parameter("auto_activate").value:
        if configured:
            node.trigger_activate()
        else:
            node.get_logger().error("Auto-activation requested, but configuration failed; skipping activation.")
    node.get_logger().info("PickLabel detector started. Spinning...")
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
