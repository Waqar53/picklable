"""The auto-labeler: one pick pair in, one scored and routed instance mask out."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import yaml

from picklabel.autolabel.align import AlignConfig, align_pair
from picklabel.autolabel.change import ChangeConfig, compute_change
from picklabel.autolabel.score import ScoreConfig, combine, decide, quality_signals
from picklabel.autolabel.segment import Candidate, grabcut_refine
from picklabel.autolabel.selector import component_at, negative_points, region_box, select_candidate
from picklabel.masks import erode
from picklabel.types import AutoLabel, PickPair, Route


@dataclass
class SegmentConfig:
    prompts: list[str] = field(default_factory=lambda: ["plastic bag", "plastic film", "plastic wrapper"])
    sam3_conf: float = 0.25
    grasp_radius_frac: float = 0.012
    region_max_dist_frac: float = 0.03
    n_negative_points: int = 4
    grabcut: bool = True


@dataclass
class PropagateConfig:
    enabled: bool = True
    change_thr: float = 0.25  # per-pixel change score that counts as "changed" between two frames
    min_visible_frac: float = 0.6  # share of a later item's mask still unchanged in the earlier frame
    min_boundary_frac: float = 0.6  # share of its rim unchanged; a shifted item fails this


def _build(cls, data: dict | None):
    data = data or {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


@dataclass
class AutoLabelConfig:
    align: AlignConfig = field(default_factory=AlignConfig)
    change: ChangeConfig = field(default_factory=ChangeConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    propagate: PropagateConfig = field(default_factory=PropagateConfig)

    @classmethod
    def from_dict(cls, data: dict) -> AutoLabelConfig:
        return cls(
            align=_build(AlignConfig, data.get("align")),
            change=_build(ChangeConfig, data.get("change")),
            segment=_build(SegmentConfig, data.get("segment")),
            score=_build(ScoreConfig, data.get("score")),
            propagate=_build(PropagateConfig, data.get("propagate")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> AutoLabelConfig:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> dict:
        return asdict(self)


class AutoLabeler:
    """Align, diff, segment, score and route one pick.

    ``text_segmenter`` (SAM 3) proposes candidates by text; ``point_segmenter`` (SAM 2.1)
    is prompted with the grasp point and the change box when no text candidate fits.
    Without either, the change region itself (snapped to edges by GrabCut) is the mask.
    """

    def __init__(self, cfg: AutoLabelConfig | None = None, text_segmenter=None, point_segmenter=None):
        self.cfg = cfg or AutoLabelConfig()
        self.text_segmenter = text_segmenter
        self.point_segmenter = point_segmenter

    def label(self, pair: PickPair, keep_debug: bool = False) -> AutoLabel:
        cfg = self.cfg
        timings: dict[str, float] = {}
        t = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal t
            now = time.perf_counter()
            timings[name] = (now - t) * 1000
            t = now

        h, w = pair.shape
        long_side = max(h, w)
        grasp = (float(np.clip(pair.grasp_xy[0], 0, w - 1)), float(np.clip(pair.grasp_xy[1], 0, h - 1)))
        radius = cfg.segment.grasp_radius_frac * long_side

        al = align_pair(pair.before, pair.after, cfg.align, pair.belt_shift_px)
        lap("align")
        ch = compute_change(pair.before, al.warped, al.valid, cfg.change)
        lap("change")
        # A removal is only observable where both frames overlap: if the belt carried the pick
        # region out of the after frame, there is nothing to compare.
        gx, gy = int(round(grasp[0])), int(round(grasp[1]))
        visible = bool(erode(al.valid, int(2 * radius))[gy, gx])
        region = component_at(ch.mask, grasp, cfg.segment.region_max_dist_frac * long_side) if visible else None

        chosen: Candidate | None = None
        details: dict[str, float] = {}
        if region is not None:
            if self.text_segmenter is not None:
                cands = self.text_segmenter(pair.before, cfg.segment.prompts)
                chosen, details = select_candidate(cands, region, ch.score, grasp, radius)
                lap("sam3")
            if chosen is None and self.point_segmenter is not None:
                negs = negative_points(region, ch.score, al.valid, cfg.segment.n_negative_points)
                cands = self.point_segmenter(pair.before, grasp, region_box(region, (h, w)), negs)
                chosen, details = select_candidate(cands, region, ch.score, grasp, radius)
                lap("sam2")
            if chosen is None:
                mask = grabcut_refine(pair.before, region) if cfg.segment.grabcut else region
                chosen = Candidate(mask=mask, score=1.0, source="change")
                lap("grabcut")

        if chosen is not None:
            signals = quality_signals(chosen.mask, region, ch.score, al.valid, grasp, radius, chosen.source,
                                      cfg.score)
            score = combine(signals, cfg.score)
        else:
            signals, score = {}, 0.0
        signals.update(details)
        if visible:
            route, score, reasons = decide(signals, score, cfg.score, region is not None, al.ok, pair.grasp_success)
        else:
            route, score, reasons = Route.HARD, 0.0, ["grasp_not_visible_after"]
        timings["total"] = sum(timings.values())

        label = AutoLabel(
            pick_id=pair.pick_id,
            session=pair.session,
            class_name=pair.class_name,
            route=route,
            score=score,
            mask=chosen.mask if chosen is not None else None,
            source=chosen.source if chosen is not None else "none",
            reasons=reasons,
            signals=signals,
            timings_ms=timings,
            align={**al.info(), "change": ch.stats},
            grasp_xy=grasp,
        )
        if keep_debug:
            label.change_score, label.change_mask = ch.score, ch.mask
            label.region, label.after_aligned = region, al.warped
        return label


def is_failed_pick(label: AutoLabel) -> bool:
    """A pick where nothing visibly left the scene (the item is still there)."""
    return label.route == Route.HARD and "no_change_at_grasp" in label.reasons
