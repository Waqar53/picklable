"""Change map: where did the scene change between the aligned before and after frames?

Two terms, both *shift-tolerant*: for every pixel, the difference is the minimum over small
displacements (up to ``shift_px``) of the after frame. Pulling one item out of wet waste
nudges its neighbours a few pixels; a nudged neighbour still finds its own texture nearby,
a removed item does not. Without this, a slid neighbour lights up across its whole area and
merges with the pick region.

* colour: CIE76 delta-E between blurred LAB images
* texture: difference of local contrast (std of lightness), which catches a textured item
  removed from a background of similar colour

Both use absolute scales rather than per-image percentiles, so a failed pick where nothing
left the scene gives an empty map instead of the top few percent of sensor noise.

Before differencing, each colour channel of the after frame is multiplied by a robust gain
(median before/after ratio over mid-tone pixels). Exposure and white balance act as
per-channel gains on the sensor, so this cancels their drift between the two shots. Grip's
ZED fork fixes exposure but still runs auto white balance. Matching LAB statistics was
tried first and left a large baseline on textured items: channel gains are not affine in LAB.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.filters import apply_hysteresis_threshold

from picklabel.masks import clean_mask, erode


@dataclass
class ChangeConfig:
    work_max_side: int = 1024
    color_scale: float = 18.0  # delta-E that counts as a full-strength colour change
    texture_scale: float = 0.5  # relative local-contrast change that counts as full strength
    w_color: float = 0.65
    w_texture: float = 0.35
    shift_px: int = 6  # displacement tolerance, in work-resolution pixels
    shift_step: int = 3
    texture_sigma: float = 2.5
    normalize_color: bool = True
    blur_sigma: float = 1.5
    low: float = 0.22
    high: float = 0.40
    open_px: int = 3
    close_px: int = 9
    min_area_frac: float = 0.0002


@dataclass
class ChangeResult:
    score: np.ndarray  # float32 in [0, 1], full resolution
    mask: np.ndarray  # bool, full resolution
    stats: dict = field(default_factory=dict)


def normalize_gain(after: np.ndarray, before: np.ndarray, valid: np.ndarray,
                   max_gain: float = 1.4) -> tuple[np.ndarray, list[float]]:
    """Per-channel gains that map the after frame's exposure / white balance onto the before frame's."""
    a = cv2.GaussianBlur(after.astype(np.float32), (0, 0), 2.0)
    b = cv2.GaussianBlur(before.astype(np.float32), (0, 0), 2.0)
    out = after.astype(np.float32)
    gains = []
    for c in range(after.shape[2]):
        ac, bc = a[..., c][valid], b[..., c][valid]
        ok = (ac > 15) & (bc > 15) & (ac < 245) & (bc < 245)
        g = float(np.clip(np.median(bc[ok] / ac[ok]), 1 / max_gain, max_gain)) if ok.sum() > 100 else 1.0
        out[..., c] *= g
        gains.append(round(g, 4))
    return np.clip(out, 0, 255).astype(np.uint8), gains


def _to_lab(img_bgr: np.ndarray, blur_sigma: float) -> np.ndarray:
    f = img_bgr.astype(np.float32) / 255.0
    if blur_sigma > 0:
        f = cv2.GaussianBlur(f, (0, 0), blur_sigma)
    return cv2.cvtColor(f, cv2.COLOR_BGR2LAB)


def _local_std(x: np.ndarray, sigma: float) -> np.ndarray:
    mu = cv2.GaussianBlur(x, (0, 0), sigma)
    return np.sqrt(np.maximum(cv2.GaussianBlur(x * x, (0, 0), sigma) - mu * mu, 0.0))


def shift_min(fb: np.ndarray, fa: np.ndarray, radius: int, step: int,
              dist: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    """Per-pixel ``min_d dist(fb(p), fa(p + d))`` over displacements ``|dx|, |dy| <= radius``."""
    if radius <= 0:
        return dist(fb, fa)
    h, w = fb.shape[:2]
    pad = ((radius, radius), (radius, radius)) + ((0, 0),) * (fa.ndim - 2)
    fa_p = np.pad(fa, pad, mode="edge")
    offsets = sorted(set(range(-radius, radius + 1, max(step, 1))) | {0, -radius, radius})
    best = None
    for dy in offsets:
        for dx in offsets:
            d = dist(fb, fa_p[radius + dy: radius + dy + h, radius + dx: radius + dx + w])
            best = d if best is None else np.minimum(best, d)
    return best


def _sq_color(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return np.einsum("ijk,ijk->ij", d, d)


def compute_change(before: np.ndarray, after_aligned: np.ndarray, valid: np.ndarray | None = None,
                   cfg: ChangeConfig | None = None) -> ChangeResult:
    cfg = cfg or ChangeConfig()
    h, w = before.shape[:2]
    if valid is None:
        valid = np.ones((h, w), bool)
    s = min(1.0, cfg.work_max_side / float(max(h, w)))
    if s < 1.0:
        size = (max(1, round(w * s)), max(1, round(h * s)))
        b = cv2.resize(before, size, interpolation=cv2.INTER_AREA)
        a = cv2.resize(after_aligned, size, interpolation=cv2.INTER_AREA)
        v = cv2.resize(valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    else:
        b, a, v = before, after_aligned, valid.astype(bool)
    v = erode(v, max(1, cfg.shift_px))

    gains: list[float] = []
    if cfg.normalize_color:
        a, gains = normalize_gain(a, b, v)
    lab_b, lab_a = _to_lab(b, cfg.blur_sigma), _to_lab(a, cfg.blur_sigma)

    delta_e = np.sqrt(shift_min(lab_b, lab_a, cfg.shift_px, cfg.shift_step, _sq_color))
    color = np.clip(delta_e / cfg.color_scale, 0.0, 1.0)

    std_b = _local_std(lab_b[..., 0], cfg.texture_sigma)
    std_a = _local_std(lab_a[..., 0], cfg.texture_sigma)
    rel = shift_min(std_b, std_a, cfg.shift_px, cfg.shift_step,
                    lambda x, y: np.abs(x - y) / (x + y + 2.0))
    texture = np.clip(rel / cfg.texture_scale, 0.0, 1.0)

    score = (cfg.w_color * color + cfg.w_texture * texture) / max(cfg.w_color + cfg.w_texture, 1e-6)
    score = cv2.GaussianBlur(score.astype(np.float32), (0, 0), 1.0)
    score[~v] = 0.0

    mask = apply_hysteresis_threshold(score, cfg.low, cfg.high) & v
    min_area = max(1, int(cfg.min_area_frac * score.size))
    mask = clean_mask(mask, cfg.open_px, cfg.close_px, min_area, fill=True) & v

    if s < 1.0:
        score_full = cv2.resize(score, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_full = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
    else:
        score_full, mask_full = score, mask
    stats = {
        "changed_frac": round(float(mask.mean()), 5),
        "score_p99": round(float(np.percentile(score[v], 99)) if v.any() else 0.0, 4),
        "valid_frac": round(float(v.mean()), 4),
        "gains_bgr": gains,
    }
    return ChangeResult(score=score_full.astype(np.float32), mask=mask_full, stats=stats)
