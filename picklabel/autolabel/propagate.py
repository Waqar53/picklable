"""Complete each frame's labels by copying later picks' masks back in time.

A pick labels one item, but the before frame shows every item still on the tray. Training
on one labelled plastic among several unlabelled ones teaches the student that plastic is
background, which costs recall. In a session where items are only removed (the home tray,
or a belt segment the robot clears), each item picked later is still present in earlier
frames, so its mask can be copied back after a frame-to-frame alignment, as long as its
pixels did not change in between (no neighbour shift). A frame is ``complete`` when every
later removal was labelled and copied back and the session ended with the tray cleared.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from picklabel.autolabel.align import AlignConfig, align_pair
from picklabel.autolabel.change import ChangeConfig, compute_change
from picklabel.autolabel.pipeline import PropagateConfig, is_failed_pick
from picklabel.masks import disk, erode, remove_small
from picklabel.types import AutoLabel, PickPair, Route


@dataclass
class Instance:
    mask: np.ndarray
    route: Route
    score: float
    source: str
    from_pick: str
    propagated: bool = False


@dataclass
class FrameLabels:
    pick_id: str
    session: str
    instances: list[Instance] = field(default_factory=list)
    complete: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def route(self) -> Route:
        r = Route.worst([i.route for i in self.instances])
        if not self.complete and r == Route.ACCEPT:
            r = Route.REVIEW  # some item in view may be unlabelled
        return r


def _own_instances(label: AutoLabel) -> list[Instance]:
    if label.mask is None or label.route == Route.HARD:
        return []
    return [Instance(label.mask, label.route, label.score, label.source, label.pick_id)]


def single_frames(labels: list[AutoLabel]) -> list[FrameLabels]:
    """No propagation (belt sessions): each frame keeps only its own pick."""
    return [FrameLabels(lb.pick_id, lb.session, _own_instances(lb), complete=False) for lb in labels]


def propagate_session(pairs: list[PickPair], labels: list[AutoLabel], cfg: PropagateConfig,
                      align_cfg: AlignConfig, change_cfg: ChangeConfig, cleared: bool) -> list[FrameLabels]:
    n = len(pairs)
    frames: list[FrameLabels] = []
    for k in range(n):
        own = labels[k]
        frame = FrameLabels(own.pick_id, own.session, _own_instances(own), complete=cleared)
        if own.route == Route.HARD and not is_failed_pick(own):
            frame.complete = False  # this item left the scene without a usable mask
            frame.notes.append(f"own pick {own.pick_id} unlabelled")
        for j in range(k + 1, n):
            later = labels[j]
            if is_failed_pick(later):
                continue  # nothing was removed at pick j
            if later.mask is None or later.route == Route.HARD:
                frame.complete = False
                frame.notes.append(f"pick {later.pick_id} unlabelled")
                continue
            # Bring frame j's before image into frame k; H maps k pixels to j pixels.
            al = align_pair(pairs[k].before, pairs[j].before, align_cfg)
            h, w = pairs[k].shape
            mask_k = cv2.warpPerspective(later.mask.astype(np.uint8), al.H, (w, h),
                                         flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
            area = int(mask_k.sum())
            if area == 0:
                frame.complete = False
                continue
            # Raw per-pixel change (no hole filling: removed neighbours can ring an untouched item).
            changed = compute_change(pairs[k].before, al.warped, al.valid, change_cfg).score >= cfg.change_thr
            visible_part = mask_k & al.valid & ~changed  # parts hidden under items removed later drop out
            inner = mask_k & ~erode(mask_k, 3)
            visible = float(visible_part.sum() / area)
            boundary = float((inner & al.valid & ~changed).sum() / max(int(inner.sum()), 1))
            if al.ok and visible >= cfg.min_visible_frac and boundary >= cfg.min_boundary_frac:
                part = cv2.morphologyEx(visible_part.astype(np.uint8), cv2.MORPH_CLOSE, disk(2)) > 0
                part = remove_small(part, max(1, int(0.05 * area)))
                frame.instances.append(Instance(part, later.route, later.score, later.source,
                                                later.pick_id, propagated=True))
            else:
                frame.complete = False
                frame.notes.append(f"pick {later.pick_id} moved or hidden (visible {visible:.2f}, "
                                   f"boundary {boundary:.2f})")
        frames.append(frame)
    return frames
