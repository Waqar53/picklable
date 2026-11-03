"""Mask candidates for the picked item.

* ``Sam3TextSegmenter``: text-prompted SAM 3 candidates, called the same way Grip's
  ``sam3_ros`` node calls it (Ultralytics ``SAM3SemanticPredictor``: ``set_image`` once,
  then one call per prompt), so thresholds tuned here transfer to that node.
* ``Sam2PointSegmenter``: SAM 2.1 prompted with the grasp point, background clicks and the
  change box. Ungated weights, so it works before SAM 3 access is approved.
* ``grabcut_refine``: model-free fallback that snaps the change region to image edges.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from picklabel.masks import dilate, erode, fill_holes, largest_component, mask_bbox, mask_iou, resize_mask


@dataclass
class Candidate:
    mask: np.ndarray  # bool HxW at full image resolution
    score: float  # model confidence in [0, 1]
    source: str  # sam3 | sam2 | change
    prompt: str = ""


def _default_half(device: str | None) -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available()) and (device is None or "cuda" in str(device))
    except ImportError:
        return False


def _results_to_candidates(results, shape: tuple[int, int], source: str, prompt: str = "") -> list[Candidate]:
    out: list[Candidate] = []
    if not results:
        return out
    res = results[0]
    if res.masks is None:
        return out
    masks = res.masks.data.cpu().numpy() > 0.5
    if res.boxes is not None and res.boxes.conf is not None and len(res.boxes.conf) == len(masks):
        scores = res.boxes.conf.cpu().numpy().astype(float)
    else:
        scores = np.ones(len(masks))
    h, w = shape
    for m, s in zip(masks, scores):
        if m.shape != (h, w):
            m = resize_mask(m, (w, h))
        if m.any() and np.isfinite(s):
            out.append(Candidate(mask=m, score=float(np.clip(s, 0.0, 1.0)), source=source, prompt=prompt))
    return out


class Sam3TextSegmenter:
    """Text-prompted SAM 3 candidates (needs ``sam3.pt`` from huggingface.co/facebook/sam3)."""

    name = "sam3"

    def __init__(self, weights: str = "sam3.pt", conf: float = 0.25, half: bool | None = None,
                 device: str | None = None):
        from ultralytics.models.sam import SAM3SemanticPredictor

        overrides = dict(conf=conf, task="segment", mode="predict", model=str(weights),
                         half=_default_half(device) if half is None else half, save=False, verbose=False)
        if device is not None:
            overrides["device"] = device
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        self.predictor.setup_model()

    def set_conf(self, conf: float) -> None:
        self.predictor.args.conf = conf

    def __call__(self, image_bgr: np.ndarray, prompts: Sequence[str]) -> list[Candidate]:
        self.predictor.set_image(image_bgr)
        out: list[Candidate] = []
        for prompt in prompts:
            out += _results_to_candidates(self.predictor(text=[prompt]), image_bgr.shape[:2], "sam3", prompt)
        return out


class Sam2PointSegmenter:
    """SAM 2.1 prompted with the grasp point (+ background clicks, + change box)."""

    name = "sam2"

    def __init__(self, weights: str = "sam2.1_t.pt", device: str | None = None):
        from ultralytics import SAM

        self.model = SAM(weights)
        self.device = device

    def __call__(self, image_bgr: np.ndarray, point_xy: tuple[float, float],
                 box_xyxy: Sequence[float] | None = None,
                 negative_xy: Sequence[tuple[float, float]] = ()) -> list[Candidate]:
        points = [[float(point_xy[0]), float(point_xy[1])]] + [[float(x), float(y)] for x, y in negative_xy]
        labels = [1] + [0] * len(negative_xy)
        kwargs = dict(points=[points], labels=[labels], verbose=False)
        if box_xyxy is not None:
            kwargs["bboxes"] = [[float(v) for v in box_xyxy]]
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.model.predict(image_bgr, **kwargs)
        return _results_to_candidates(results, image_bgr.shape[:2], "sam2")


def grabcut_refine(image_bgr: np.ndarray, region: np.ndarray, iters: int = 3,
                   min_iou: float = 0.6) -> np.ndarray:
    """Snap a change region to the item's edges in the before image.

    The eroded core is definite foreground, a ring around the region is probable
    background and everything further out is definite background. If GrabCut wanders
    off (IoU with the region below ``min_iou``), the region is returned unchanged.
    """
    area = int(region.sum())
    if area < 50:
        return region
    r = max(3, int(0.12 * np.sqrt(area)))
    box = mask_bbox(dilate(region, 3 * r))
    if box is None:
        return region
    x1, y1, x2, y2 = box
    crop = np.ascontiguousarray(image_bgr[y1:y2, x1:x2])
    reg = region[y1:y2, x1:x2]
    gc = np.full(reg.shape, cv2.GC_BGD, np.uint8)
    gc[dilate(reg, 2 * r)] = cv2.GC_PR_BGD
    gc[reg] = cv2.GC_PR_FGD
    core = erode(reg, r)
    gc[core] = cv2.GC_FGD
    if not (gc == cv2.GC_BGD).any() and not (gc == cv2.GC_PR_BGD).any():
        return region
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return region
    fg = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    fg = fill_holes(largest_component(fg))
    refined = np.zeros_like(region, bool)
    refined[y1:y2, x1:x2] = fg
    return refined if mask_iou(refined, region) >= min_iou else region
