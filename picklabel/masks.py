"""Binary-mask utilities: IoU, boxes, COCO RLE, polygons, NMS, morphology.

Only NumPy and OpenCV, so the auto-labeler also runs in the browser (Pyodide).
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight box ``(x1, y1, x2, y2)`` with exclusive max, or None for an empty mask."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 0.0


def iou_matrix(masks_a: Sequence[np.ndarray], masks_b: Sequence[np.ndarray]) -> np.ndarray:
    """Pairwise IoU; intersections are only counted inside overlapping boxes."""
    out = np.zeros((len(masks_a), len(masks_b)), np.float32)
    boxes_b = [mask_bbox(m) for m in masks_b]
    areas_b = [int(np.count_nonzero(m)) for m in masks_b]
    for i, ma in enumerate(masks_a):
        ba = mask_bbox(ma)
        if ba is None:
            continue
        area_a = int(np.count_nonzero(ma))
        for j, (mb, bb) in enumerate(zip(masks_b, boxes_b)):
            if bb is None:
                continue
            x1, y1 = max(ba[0], bb[0]), max(ba[1], bb[1])
            x2, y2 = min(ba[2], bb[2]), min(ba[3], bb[3])
            if x2 <= x1 or y2 <= y1:
                continue
            inter = np.count_nonzero(ma[y1:y2, x1:x2] & mb[y1:y2, x1:x2])
            if inter:
                out[i, j] = inter / (area_a + areas_b[j] - inter)
    return out


def mask_nms(masks: Sequence[np.ndarray], scores: Sequence[float], iou_thr: float = 0.6) -> list[int]:
    """Greedy mask NMS. Returns the indices kept, highest score first."""
    if len(masks) == 0:
        return []
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    ious = iou_matrix(masks, masks)
    keep: list[int] = []
    for i in order:
        if all(ious[i, k] < iou_thr for k in keep):
            keep.append(int(i))
    return keep


# --- COCO run-length encoding (uncompressed counts, column-major like pycocotools) ---


def rle_encode(mask: np.ndarray) -> dict:
    h, w = mask.shape
    flat = mask.reshape(-1, order="F").astype(bool)
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate([[0], change, [flat.size]])).tolist()
    if flat.size and flat[0]:
        counts = [0, *counts]
    return {"size": [int(h), int(w)], "counts": [int(c) for c in counts]}


def rle_decode(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    counts = np.asarray(rle["counts"], dtype=np.int64)
    values = (np.arange(counts.size) % 2).astype(bool)
    flat = np.repeat(values, counts)
    if flat.size != h * w:
        raise ValueError(f"RLE counts sum to {flat.size}, expected {h * w}")
    return flat.reshape((h, w), order="F")


# --- polygons ---


def mask_to_polygons(mask: np.ndarray, min_area: float = 16.0, epsilon: float = 1.0) -> list[list[float]]:
    """Outer contours as flat ``[x1, y1, x2, y2, ...]`` lists, largest first."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < min_area:
            continue
        if epsilon > 0:
            c = cv2.approxPolyDP(c, epsilon, True)
        if len(c) >= 3:
            polys.append(c.reshape(-1).astype(float).tolist())
    return polys


def polygons_to_mask(polys: Sequence[Sequence[float]], height: int, width: int) -> np.ndarray:
    canvas = np.zeros((height, width), np.uint8)
    pts = [np.asarray(p, np.float64).reshape(-1, 2).round().astype(np.int32) for p in polys if len(p) >= 6]
    if pts:
        cv2.fillPoly(canvas, pts, 1)
    return canvas.astype(bool)


# --- morphology ---


def disk(radius: int) -> np.ndarray:
    radius = max(int(radius), 0)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    return cv2.dilate(mask.astype(np.uint8), disk(radius)).astype(bool)


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    return cv2.erode(mask.astype(np.uint8), disk(radius)).astype(bool)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Same result as ``scipy.ndimage.binary_fill_holes``: background not 4-connected to the border is a hole."""
    background = np.pad((~mask.astype(bool)).astype(np.uint8), 1, constant_values=1)
    flood = np.zeros((background.shape[0] + 2, background.shape[1] + 2), np.uint8)
    cv2.floodFill(background, flood, (0, 0), 0, flags=4)
    return mask.astype(bool) | background[1:-1, 1:-1].astype(bool)


def hysteresis(score: np.ndarray, low: float, high: float) -> np.ndarray:
    """Same result as ``skimage.filters.apply_hysteresis_threshold``: 4-connected regions above ``low``
    that contain a pixel above ``high``."""
    n, labels = cv2.connectedComponents((score > low).astype(np.uint8), connectivity=4)
    keep = np.zeros(n, bool)
    keep[np.unique(labels[score > high])] = True
    keep[0] = False
    return keep[labels]


def remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, bool)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def clean_mask(mask: np.ndarray, open_px: int = 0, close_px: int = 0, min_area: int = 0, fill: bool = True) -> np.ndarray:
    m = mask.astype(np.uint8)
    if open_px > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, disk(open_px))
    if close_px > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, disk(close_px))
    out = m.astype(bool)
    if fill:
        out = fill_holes(out)
    if min_area > 0:
        out = remove_small(out, min_area)
    return out


def resize_mask(mask: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    if (mask.shape[1], mask.shape[0]) == tuple(size_wh):
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), size_wh, interpolation=cv2.INTER_NEAREST).astype(bool)


# --- instance-id maps ---


def instances_from_id_map(id_map: np.ndarray) -> list[np.ndarray]:
    ids = [int(i) for i in np.unique(id_map) if i != 0]
    return [id_map == i for i in ids]


def id_map_from_instances(masks: Sequence[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, np.uint16)
    for i, m in enumerate(masks, start=1):
        out[m & (out == 0)] = i
    return out
