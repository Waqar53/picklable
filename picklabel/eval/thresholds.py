"""Per-class and per-size thresholds from a prediction cache.

One global threshold is a compromise: small fragments score lower than whole bags, so the
threshold that keeps bags precise drops fragments. For each group this picks the lowest
threshold with precision >= target, and reports the recall it buys. The output YAML plugs
into the ROS 2 node (``class_thresholds``) or into tiled inference.

    picklabel thresholds --preds report/eval/R3/preds.jsonl --gt data/gold --target 0.9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from picklabel.eval.data import load_gt, load_predictions
from picklabel.eval.metrics import SIZE_BUCKETS, match_image


def _bucket(area: int) -> str:
    for name, (lo, hi) in SIZE_BUCKETS.items():
        if lo <= area < hi:
            return name
    return "large"


def lowest_threshold(scores: np.ndarray, tp: np.ndarray, n_gt: int, target: float) -> tuple[float | None, float]:
    order = np.argsort(-scores, kind="stable")
    s, t = scores[order], tp[order]
    ctp = np.cumsum(t)
    prec = ctp / np.arange(1, len(t) + 1)
    ok = np.flatnonzero(prec >= target)
    if ok.size == 0:
        return None, 0.0
    i = int(ok[-1])  # deepest point that still meets the target
    return float(s[i]), float(ctp[i] / max(n_gt, 1))


def per_group(preds: list[dict], gts_by_image: dict, by: str, target: float) -> dict:
    groups: dict[str, dict[str, list]] = {}
    gt_counts: dict[str, int] = {}
    for rec in preds:
        gt = gts_by_image.get(rec["image"], [])
        ev = match_image(rec["masks"], rec["scores"], gt)
        for i, m in enumerate(rec["masks"]):
            key = rec["labels"][i] if by == "class" else _bucket(int(m.sum()))
            g = groups.setdefault(key, {"scores": [], "tp": []})
            g["scores"].append(rec["scores"][i])
            g["tp"].append(bool(ev.tp[i]))
        for area in ev.gt_areas:
            # Ground truth is class-agnostic ("plastic"), so per-class recall is against all of it.
            key = "__all__" if by == "class" else _bucket(int(area))
            gt_counts[key] = gt_counts.get(key, 0) + 1
    out = {}
    for key, g in groups.items():
        n_gt = gt_counts.get("__all__" if by == "class" else key, 0)
        thr, recall = lowest_threshold(np.array(g["scores"]), np.array(g["tp"]), n_gt, target)
        out[key] = {"threshold": None if thr is None else round(thr, 4), "recall": round(recall, 4),
                    "n_pred": len(g["scores"]), "n_gt": n_gt}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Lowest threshold per class / size bucket meeting a precision target.")
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--out", default=None, help="YAML output (default: next to --preds)")
    args = ap.parse_args(argv)
    preds = list(load_predictions(args.preds))
    gts_by_image = {str(g.image_path): g.masks for g in load_gt(args.gt, args.image_dir)}
    result = {
        "target_precision": args.target,
        "size_thresholds": per_group(preds, gts_by_image, "size", args.target),
        "class_thresholds": per_group(preds, gts_by_image, "class", args.target),
    }
    out = Path(args.out) if args.out else Path(args.preds).with_name("thresholds.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(result, f, sort_keys=False)
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
