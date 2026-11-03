"""Find the change component and the mask candidate that belong to the grasp."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from picklabel.autolabel.segment import Candidate
from picklabel.masks import dilate, mask_bbox, mask_iou


def _pixel(xy: tuple[float, float], shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    return min(max(int(round(xy[0])), 0), w - 1), min(max(int(round(xy[1])), 0), h - 1)


def near_point(mask: np.ndarray, xy: tuple[float, float], radius: float) -> bool:
    """True if ``mask`` has a pixel within ``radius`` of ``xy``."""
    x, y = _pixel(xy, mask.shape)
    if mask[y, x]:
        return True
    r = int(np.ceil(radius))
    if r <= 0:
        return False
    y1, y2, x1, x2 = max(0, y - r), min(mask.shape[0], y + r + 1), max(0, x - r), min(mask.shape[1], x + r + 1)
    ys, xs = np.nonzero(mask[y1:y2, x1:x2])
    if ys.size == 0:
        return False
    return bool(((xs + x1 - x) ** 2 + (ys + y1 - y) ** 2).min() <= radius**2)


def component_at(change_mask: np.ndarray, xy: tuple[float, float], max_dist_px: float) -> np.ndarray | None:
    """The change component under the grasp, else the nearest one within ``max_dist_px``.

    Keeping only this component discards neighbours that shifted when the item was pulled.
    """
    n, labels = cv2.connectedComponents(change_mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    x, y = _pixel(xy, change_mask.shape)
    cid = int(labels[y, x])
    if cid == 0:
        ys, xs = np.nonzero(labels)
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        i = int(np.argmin(d2))
        if d2[i] > max_dist_px**2:
            return None
        cid = int(labels[ys[i], xs[i]])
    return labels == cid


def expand_box(box: tuple[int, int, int, int], frac: float, shape: tuple[int, int]) -> list[float]:
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * frac, (y2 - y1) * frac
    h, w = shape
    return [max(0.0, x1 - dx), max(0.0, y1 - dy), min(w - 1.0, x2 + dx), min(h - 1.0, y2 + dy)]


def negative_points(region: np.ndarray, change_score: np.ndarray, valid: np.ndarray, n: int = 4,
                    rng: np.random.Generator | None = None) -> list[tuple[float, float]]:
    """Background clicks for SAM: unchanged pixels on a ring around the removed item.

    They often land on touching neighbours, which is exactly where SAM tends to leak.
    """
    if n <= 0:
        return []
    rng = rng or np.random.default_rng(0)
    area = float(region.sum())
    radius = np.sqrt(area / np.pi)
    ring = dilate(region, int(1.6 * radius) + 3) & ~dilate(region, int(0.5 * radius) + 3)
    ring &= valid & (change_score < 0.1)
    ys, xs = np.nonzero(ring)
    if ys.size < n:
        return []
    cy, cx = np.argwhere(region).mean(axis=0)
    sector = ((np.arctan2(ys - cy, xs - cx) + np.pi) / (2 * np.pi) * n).astype(int) % n
    points = []
    for s in range(n):
        idx = np.flatnonzero(sector == s)
        if idx.size:
            i = int(rng.choice(idx))
            points.append((float(xs[i]), float(ys[i])))
    return points


def select_candidate(candidates: Sequence[Candidate], region: np.ndarray | None, change_score: np.ndarray,
                     grasp_xy: tuple[float, float], grasp_radius_px: float,
                     max_area_ratio: float = 4.0) -> tuple[Candidate | None, dict]:
    """Choose the candidate at the grasp that best explains the change region."""
    best, best_s = None, -1.0
    region_area = float(region.sum()) if region is not None else 0.0
    for c in candidates:
        if not near_point(c.mask, grasp_xy, grasp_radius_px):
            continue
        if region is not None and c.mask.sum() > max_area_ratio * region_area:
            continue  # e.g. one SAM 3 mask covering a whole pile of film
        agreement = mask_iou(c.mask, region) if region is not None else 0.0
        support = float(change_score[c.mask].mean()) if c.mask.any() else 0.0
        s = 0.5 * agreement + 0.3 * min(support / 0.5, 1.0) + 0.2 * c.score
        if s > best_s:
            best, best_s = c, s
    details = {"n_candidates": float(len(candidates)), "selection": round(best_s, 4) if best else 0.0}
    return best, details


def region_box(region: np.ndarray, shape: tuple[int, int], frac: float = 0.1) -> list[float] | None:
    box = mask_bbox(region)
    return expand_box(box, frac, shape) if box is not None else None
