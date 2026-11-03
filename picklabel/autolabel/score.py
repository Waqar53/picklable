"""Quality score for an auto-label, and the routing decision built on it.

Every signal is in [0, 1]. The weights and cutoffs are starting values: calibrate them on
the gold set (``picklabel eval-autolabel``) so accepted labels average IoU >= 0.8.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from picklabel.autolabel.selector import near_point
from picklabel.masks import dilate, mask_iou
from picklabel.types import Route


@dataclass
class ScoreConfig:
    accept: float = 0.70
    review: float = 0.40
    support_ref: float = 0.5
    core_thr: float = 0.40  # change score that counts as "strong" (the hysteresis high threshold)
    min_area_frac: float = 0.0002
    max_area_frac: float = 0.25
    weights: dict[str, float] = field(default_factory=lambda: {
        "agreement": 0.25, "core": 0.20, "support": 0.15, "contrast": 0.15, "shape": 0.10, "source": 0.15,
    })
    source_prior: dict[str, float] = field(default_factory=lambda: {"sam3": 1.0, "sam2": 0.85, "change": 0.6})


def _solidity_and_fragments(mask: np.ndarray) -> tuple[float, float, int]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return 0.0, 0.0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    frag = float(areas.max() / areas.sum())
    contours, _ = cv2.findContours((labels == largest).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    hull_area = cv2.contourArea(cv2.convexHull(np.vstack(contours))) if contours else 0.0
    solidity = float(areas.max() / hull_area) if hull_area > 0 else 0.0
    return min(solidity, 1.0), frag, n - 1


def quality_signals(mask: np.ndarray, region: np.ndarray | None, change_score: np.ndarray,
                    valid: np.ndarray, grasp_xy: tuple[float, float], grasp_radius_px: float,
                    source: str, cfg: ScoreConfig) -> dict[str, float]:
    area = int(mask.sum())
    if area == 0:
        return {"agreement": 0.0, "core": 0.0, "support": 0.0, "contrast": 0.0, "shape": 0.0, "source": 0.0,
                "grasp": 0.0}
    agreement = mask_iou(mask, region) if region is not None else 0.0
    inside = change_score[mask]
    mean_in = float(inside.mean())
    core = float((inside >= cfg.core_thr).mean())  # low when the mask runs into weakly-changed neighbours
    ring = dilate(mask, max(3, int(0.15 * np.sqrt(area)))) & ~mask & valid
    mean_ring = float(change_score[ring].mean()) if ring.any() else 0.0
    area_frac = area / mask.size
    solidity, frag, n_parts = _solidity_and_fragments(mask)
    area_ok = 1.0 if cfg.min_area_frac <= area_frac <= cfg.max_area_frac else 0.0
    gx, gy = min(max(int(round(grasp_xy[0])), 0), mask.shape[1] - 1), min(max(int(round(grasp_xy[1])), 0),
                                                                          mask.shape[0] - 1)
    grasp = 1.0 if mask[gy, gx] else (0.5 if near_point(mask, grasp_xy, grasp_radius_px) else 0.0)
    return {
        "agreement": agreement,
        "core": core,
        "support": float(np.clip(mean_in / cfg.support_ref, 0.0, 1.0)),
        "contrast": float(np.clip((mean_in - mean_ring) / cfg.support_ref, 0.0, 1.0)),
        "shape": area_ok * frag * float(np.clip(solidity / 0.5, 0.0, 1.0)),
        "source": cfg.source_prior.get(source, 0.5),
        "grasp": grasp,
        "area_frac": area_frac,
        "solidity": solidity,
        "parts": float(n_parts),
    }


def combine(signals: dict[str, float], cfg: ScoreConfig) -> float:
    total = sum(cfg.weights.values())
    return float(sum(w * signals.get(k, 0.0) for k, w in cfg.weights.items()) / total) if total else 0.0


def decide(signals: dict[str, float], score: float, cfg: ScoreConfig, region_found: bool,
           align_ok: bool, grasp_success: bool | None) -> tuple[Route, float, list[str]]:
    """Route a label. Gripper feedback, when present, overrides what the pixels suggest."""
    reasons: list[str] = []
    if not region_found:
        reasons.append("no_change_at_grasp")
        if grasp_success is False:
            reasons.append("gripper_empty")  # missed pick: a likely perception false positive
        elif grasp_success is True:
            reasons.append("gripper_holding_but_no_visible_change")  # transparent / occluded item
        return Route.HARD, 0.0, reasons

    if signals.get("grasp", 0.0) == 0.0:
        reasons.append("grasp_outside_mask")
        score = min(score, cfg.review - 1e-3)
    elif signals.get("grasp", 0.0) < 1.0:
        reasons.append("grasp_on_edge")
        score *= 0.95
    if not align_ok:
        reasons.append("alignment_suspect")
        score *= 0.75
    if grasp_success is False:
        reasons.append("gripper_empty_but_scene_changed")  # pushed, not picked
        score = min(score, cfg.accept - 1e-3)
    if signals.get("support", 0.0) < 0.4:
        reasons.append("weak_change")
    if signals.get("agreement", 0.0) < 0.5:
        reasons.append("mask_disagrees_with_change")
    if signals.get("shape", 0.0) < 0.5:
        reasons.append("implausible_shape")

    if score >= cfg.accept:
        route = Route.ACCEPT
    elif score >= cfg.review:
        route = Route.REVIEW
    else:
        route = Route.HARD
    return route, float(score), reasons
