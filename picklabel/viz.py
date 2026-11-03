"""Small drawing helpers for previews, galleries and the README GIF."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from picklabel.types import AutoLabel, PickPair, Route

ROUTE_BGR = {Route.ACCEPT: (90, 200, 70), Route.REVIEW: (0, 190, 255), Route.HARD: (70, 70, 235)}


def heatmap(score: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap((np.clip(score, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def overlay(image: np.ndarray, mask: np.ndarray | None, color=(0, 220, 255), alpha: float = 0.45,
            contour: bool = True) -> np.ndarray:
    out = image.copy()
    if mask is None or not mask.any():
        return out
    tint = np.zeros_like(out)
    tint[:] = color
    out[mask] = (out[mask] * (1 - alpha) + tint[mask] * alpha).astype(np.uint8)
    if contour:
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, color, max(1, image.shape[1] // 400), cv2.LINE_AA)
    return out


def draw_point(image: np.ndarray, xy: Sequence[float] | None, color=(60, 255, 60)) -> np.ndarray:
    out = image.copy()
    if xy is None:
        return out
    r = max(6, image.shape[1] // 90)
    p = (int(round(xy[0])), int(round(xy[1])))
    cv2.circle(out, p, r, (0, 0, 0), max(2, r // 3) + 2, cv2.LINE_AA)
    cv2.circle(out, p, r, color, max(2, r // 3), cv2.LINE_AA)
    cv2.drawMarker(out, p, color, cv2.MARKER_CROSS, r, 1, cv2.LINE_AA)
    return out


def caption(image: np.ndarray, text: str, color=(255, 255, 255), bar=(25, 25, 25)) -> np.ndarray:
    h, w = image.shape[:2]
    bar_h = max(22, h // 16)
    out = np.vstack([np.full((bar_h, w, 3), bar, np.uint8), image])
    scale = bar_h / 40
    cv2.putText(out, text, (8, int(bar_h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                max(1, int(bar_h / 20)), cv2.LINE_AA)
    return out


def hstack(images: Sequence[np.ndarray], height: int, gap: int = 6) -> np.ndarray:
    tiles = []
    for img in images:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        s = height / img.shape[0]
        tiles.append(cv2.resize(img, (max(1, round(img.shape[1] * s)), height), interpolation=cv2.INTER_AREA))
        tiles.append(np.full((height, gap, 3), 255, np.uint8))
    return np.hstack(tiles[:-1]) if tiles else np.zeros((height, 1, 3), np.uint8)


def autolabel_panel(pair: PickPair, label: AutoLabel, height: int = 300) -> np.ndarray:
    """before + grasp | after (aligned) | change map + region | mask, score and route."""
    before = draw_point(pair.before, label.grasp_xy)
    after = label.after_aligned if label.after_aligned is not None else pair.after
    change = heatmap(label.change_score) if label.change_score is not None else np.zeros_like(before)
    if label.region is not None:
        cs, _ = cv2.findContours(label.region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(change, cs, -1, (255, 255, 255), max(1, change.shape[1] // 300), cv2.LINE_AA)
    color = ROUTE_BGR[label.route]
    result = draw_point(overlay(pair.before, label.mask, color), label.grasp_xy)
    tiles = [
        caption(before, "before + grasp"),
        caption(after, "after (aligned)"),
        caption(change, "change map"),
        caption(result, f"{label.route.value.upper()}  q={label.score:.2f}  {label.source}", color=color),
    ]
    return hstack(tiles, height)
