"""Instance-mask metrics for a picking robot.

Predictions are matched to ground truth greedily by mask IoU, highest score first, per
image. Sweeping the score threshold gives the precision/recall curve. A picking robot
cares about one operating point more than the area under the curve: the recall it gets
at a precision high enough that false picks don't waste cycles. So recall @ P>=0.90 and
the threshold that achieves it come first in every report; AP is there for comparison.

Duplicates (a second mask on an already-matched item) are counted separately. They are
what a multi-prompt detector without cross-prompt NMS produces, and each one is a wasted
pick.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from picklabel.masks import iou_matrix

SIZE_BUCKETS = {"small": (0, 32**2), "medium": (32**2, 96**2), "large": (96**2, float("inf"))}


@dataclass
class ImageEval:
    scores: np.ndarray  # per prediction
    tp: np.ndarray  # per prediction, bool
    duplicate: np.ndarray  # per prediction, bool: matched an already-taken GT
    gt_areas: np.ndarray  # per GT
    gt_best_score: np.ndarray  # per GT: score of the prediction that matched it (-1 if none)


def match_image(pred_masks: Sequence[np.ndarray], pred_scores: Sequence[float], gt_masks: Sequence[np.ndarray],
                iou_thr: float = 0.5) -> ImageEval:
    scores = np.asarray(pred_scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    n_p, n_g = len(pred_masks), len(gt_masks)
    tp = np.zeros(n_p, bool)
    dup = np.zeros(n_p, bool)
    gt_best = np.full(n_g, -1.0)
    if n_p and n_g:
        ious = iou_matrix(pred_masks, gt_masks)
        taken = np.zeros(n_g, bool)
        for i in order:
            free = np.where(~taken, ious[i], -1.0)
            j = int(np.argmax(free))
            if free[j] >= iou_thr:
                taken[j] = True
                tp[i] = True
                gt_best[j] = scores[i]
            elif ious[i].max() >= iou_thr:
                dup[i] = True
    gt_areas = np.array([int(m.sum()) for m in gt_masks], dtype=np.int64)
    return ImageEval(scores=scores, tp=tp, duplicate=dup, gt_areas=gt_areas, gt_best_score=gt_best)


@dataclass
class PRCurve:
    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    n_gt: int

    def at(self, thr: float) -> tuple[float, float]:
        """Precision and recall when keeping predictions with score >= thr."""
        keep = self.thresholds >= thr
        if not keep.any():
            return 1.0, 0.0
        i = int(np.flatnonzero(keep)[-1])
        return float(self.precision[i]), float(self.recall[i])


def pr_curve(evals: Sequence[ImageEval]) -> PRCurve:
    scores = np.concatenate([e.scores for e in evals]) if evals else np.zeros(0)
    tps = np.concatenate([e.tp for e in evals]) if evals else np.zeros(0, bool)
    n_gt = int(sum(len(e.gt_areas) for e in evals))
    order = np.argsort(-scores, kind="stable")
    scores, tps = scores[order], tps[order]
    ctp = np.cumsum(tps)
    cfp = np.cumsum(~tps)
    precision = ctp / np.maximum(ctp + cfp, 1)
    recall = ctp / max(n_gt, 1)
    return PRCurve(thresholds=scores, precision=precision, recall=recall, n_gt=n_gt)


def average_precision(curve: PRCurve) -> float:
    """COCO-style 101-point interpolated AP."""
    if curve.n_gt == 0 or curve.recall.size == 0:
        return 0.0
    prec = np.maximum.accumulate(curve.precision[::-1])[::-1]
    out = []
    for r in np.linspace(0, 1, 101):
        idx = np.searchsorted(curve.recall, r, side="left")
        out.append(prec[idx] if idx < prec.size else 0.0)
    return float(np.mean(out))


def recall_at_precision(curve: PRCurve, min_precision: float = 0.9) -> tuple[float, float | None]:
    """Best recall with precision >= ``min_precision``, and the score threshold that gives it."""
    ok = curve.precision >= min_precision
    if not ok.any():
        return 0.0, None
    i = int(np.argmax(np.where(ok, curve.recall, -1)))
    return float(curve.recall[i]), float(curve.thresholds[i])


def best_f1(curve: PRCurve) -> tuple[float, float | None]:
    if curve.recall.size == 0:
        return 0.0, None
    f1 = 2 * curve.precision * curve.recall / np.maximum(curve.precision + curve.recall, 1e-9)
    i = int(np.argmax(f1))
    return float(f1[i]), float(curve.thresholds[i])


def recall_by_size(evals: Sequence[ImageEval], thr: float) -> dict[str, dict[str, float]]:
    areas = np.concatenate([e.gt_areas for e in evals]) if evals else np.zeros(0)
    best = np.concatenate([e.gt_best_score for e in evals]) if evals else np.zeros(0)
    out = {}
    for name, (lo, hi) in SIZE_BUCKETS.items():
        sel = (areas >= lo) & (areas < hi)
        if sel.any():
            out[name] = {"n": int(sel.sum()), "recall": round(float((best[sel] >= thr).mean()), 4)}
    return out


@dataclass
class Report:
    n_images: int
    n_gt: int
    n_pred: int
    ap50: float
    ap50_95: float
    recall_at_p90: float
    thr_at_p90: float | None
    best_f1: float
    thr_best_f1: float | None
    at_threshold: dict[str, float] = field(default_factory=dict)
    by_size: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def evaluate(preds: Sequence[tuple[Sequence[np.ndarray], Sequence[float]]], gts: Sequence[Sequence[np.ndarray]],
             operating_thr: float | None = None, min_precision: float = 0.9) -> tuple[Report, PRCurve]:
    """``preds[i] = (masks, scores)`` and ``gts[i] = masks`` for image ``i``."""
    evals50 = [match_image(pm, ps, g, 0.5) for (pm, ps), g in zip(preds, gts)]
    curve = pr_curve(evals50)
    ap50 = average_precision(curve)
    aps = [ap50] + [
        average_precision(pr_curve([match_image(pm, ps, g, t) for (pm, ps), g in zip(preds, gts)]))
        for t in np.arange(0.55, 0.96, 0.05)
    ]
    r90, t90 = recall_at_precision(curve, min_precision)
    f1, tf1 = best_f1(curve)
    thr = operating_thr if operating_thr is not None else (t90 if t90 is not None else (tf1 or 0.5))
    p, r = curve.at(thr)
    kept = [e.scores >= thr for e in evals50]
    dups = int(sum((e.duplicate & k).sum() for e, k in zip(evals50, kept)))
    fps = int(sum((~e.tp & k).sum() for e, k in zip(evals50, kept)))
    report = Report(
        n_images=len(gts), n_gt=curve.n_gt, n_pred=int(curve.thresholds.size),
        ap50=ap50, ap50_95=float(np.mean(aps)), recall_at_p90=r90, thr_at_p90=t90,
        best_f1=f1, thr_best_f1=tf1,
        at_threshold={"threshold": round(float(thr), 4), "precision": round(p, 4), "recall": round(r, 4),
                      "false_positives": fps, "duplicates": dups},
        by_size=recall_by_size(evals50, thr),
    )
    return report, curve
