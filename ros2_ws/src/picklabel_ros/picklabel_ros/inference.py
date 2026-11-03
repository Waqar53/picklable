"""ROS-free core of the detector node: student inference -> detections, plus drawing.

Kept free of rclpy so it runs and is tested without ROS. The output conventions follow
Grip's sam3_ros node: ``instance_id = f"{class}_{i}"`` and the box is the bounding
rectangle of the mask pixels.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    class_name: str
    score: float
    mask: np.ndarray  # bool HxW, image resolution
    bbox_xywh: tuple[int, int, int, int]
    instance_id: str


def parse_prompt_text(raw) -> list[str]:
    """Same parsing rules as sam3_ros (string array, "['a', 'b']", or "a, b")."""
    if isinstance(raw, (list, tuple)):
        return [str(v).strip() for v in raw if str(v).strip()]
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except (ValueError, SyntaxError):
            pass
        if value.startswith("[") and value.endswith("]"):
            return [t.strip().strip("'\"") for t in value[1:-1].split(",") if t.strip().strip("'\"")]
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def parse_class_thresholds(values) -> dict[str, float]:
    """``["plastic=0.45", "film=0.38"]`` -> ``{"plastic": 0.45, "film": 0.38}``."""
    out = {}
    for item in parse_prompt_text(values):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = float(v)
    return out


def results_to_detections(result, image_shape: tuple[int, int], threshold: float,
                          class_filter: list[str] | None = None,
                          class_thresholds: dict[str, float] | None = None) -> list[Detection]:
    """Convert one Ultralytics segmentation result into detections above threshold."""
    if result.masks is None or result.boxes is None or len(result.masks.data) == 0:
        return []
    h, w = image_shape
    masks = result.masks.data.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    names = result.names
    keep_classes = set(class_filter) if class_filter else None
    out: list[Detection] = []
    order = np.argsort(-scores, kind="stable")
    for i in order:
        name = names[int(classes[i])] if isinstance(names, dict) else names[int(classes[i])]
        score = float(np.clip(scores[i], 0.0, 1.0))
        if not np.isfinite(score):
            continue
        if keep_classes is not None and name not in keep_classes:
            continue
        thr = (class_thresholds or {}).get(name, threshold)
        if score < thr:
            continue
        m = masks[i] > 0.5
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        x, y, bw, bh = cv2.boundingRect(np.column_stack((xs, ys)).astype(np.int32))
        out.append(Detection(name, score, m, (int(x), int(y), int(bw), int(bh)), f"{name}_{len(out)}"))
    return out


def color_for(name: str) -> tuple[int, int, int]:
    rng = sum(ord(c) * 131 ** i for i, c in enumerate(name)) % (256 ** 3)  # stable across runs, unlike hash()
    return ((rng >> 16) & 255, (rng >> 8) & 255, rng & 255)


def draw(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    overlay = image.copy()
    for d in detections:
        color = color_for(d.class_name)
        overlay[d.mask] = (0.6 * overlay[d.mask] + 0.4 * np.array(color)).astype(np.uint8)
        x, y, w, h = d.bbox_xywh
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.putText(overlay, f"{d.class_name} {d.score:.2f}", (x, max(y - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2, cv2.LINE_AA)
    return overlay


class StudentDetector:
    """Thin wrapper over an Ultralytics YOLO-seg model (.pt, .onnx or TensorRT .engine)."""

    def __init__(self, weights: str, device: str | None = None, half: bool = False, imgsz: int = 1024,
                 max_det: int = 100):
        from ultralytics import YOLO

        self.model = YOLO(weights, task="segment")
        self.device, self.half, self.imgsz, self.max_det = device, half, imgsz, max_det

    def __call__(self, image_bgr: np.ndarray, threshold: float, class_filter: list[str] | None = None,
                 class_thresholds: dict[str, float] | None = None) -> list[Detection]:
        floor = min([threshold, *(class_thresholds or {}).values()])
        result = self.model.predict(image_bgr, imgsz=self.imgsz, conf=floor, device=self.device, half=self.half,
                                    retina_masks=True, max_det=self.max_det, verbose=False)[0]
        return results_to_detections(result, image_bgr.shape[:2], threshold, class_filter, class_thresholds)
