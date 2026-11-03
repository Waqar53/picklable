"""How good are the auto-labels, and where should the routing cutoffs sit?

Reads an auto-label run whose picks had gold masks (``gt_mask.png``: the hand-labelled gold
set, or synthetic data) and reports mask IoU by route and source. It then picks cutoffs from
the data instead of by hand:

* ``accept``: the lowest score at which the labels scoring at or above it reach the target
  mean IoU and the target share of IoU >= 0.75 (so accepted labels can go straight into
  training);
* ``review``: the lowest score at which labels between it and ``accept`` still average
  IoU >= 0.5 (fixing them is faster than drawing from scratch).

    picklabel eval-autolabel runs/autolabel/gold --out report/autolabel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from picklabel.eval.plots import calibration_figure
from picklabel.io import write_json


def load_records(run_dir: str | Path) -> list[dict]:
    with open(Path(run_dir) / "manifest.jsonl") as f:
        recs = [json.loads(line) for line in f]
    return [r for r in recs if r.get("gold_iou") is not None and not r.get("failed_pick")]


def calibrate(scores: np.ndarray, ious: np.ndarray, target_mean_iou: float = 0.85, good_iou: float = 0.75,
              target_good_frac: float = 0.9, review_min_iou: float = 0.5) -> tuple[float, float]:
    candidates = np.unique(np.round(scores, 3))
    accept = 1.0
    for t in candidates:  # ascending: the first passing cutoff accepts the most labels
        sel = scores >= t
        if sel.sum() >= 3 and ious[sel].mean() >= target_mean_iou and (ious[sel] >= good_iou).mean() >= target_good_frac:
            accept = float(t)
            break
    review = accept
    for t in candidates[candidates < accept]:
        sel = (scores >= t) & (scores < accept)
        if sel.any() and ious[sel].mean() >= review_min_iou:
            review = float(t)
            break
    return accept, review


def summarize(recs: list[dict]) -> dict:
    out: dict = {"n": len(recs)}
    ious = np.array([r["gold_iou"] for r in recs])
    out["mean_iou"] = round(float(ious.mean()), 4) if len(ious) else None
    for key in ("route", "source"):
        groups = {}
        for name in sorted({r[key] for r in recs}):
            v = np.array([r["gold_iou"] for r in recs if r[key] == name])
            groups[name] = {"n": int(v.size), "mean_iou": round(float(v.mean()), 4),
                            "iou>=0.75": round(float((v >= 0.75).mean()), 4)}
        out[f"by_{key}"] = groups
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Gold-set IoU of auto-labels and data-driven routing cutoffs.")
    ap.add_argument("run", help="an auto-label run folder (with manifest.jsonl)")
    ap.add_argument("--out", default="report/autolabel")
    ap.add_argument("--target-mean-iou", type=float, default=0.85)
    ap.add_argument("--target-good-frac", type=float, default=0.9)
    args = ap.parse_args(argv)

    recs = load_records(args.run)
    if not recs:
        raise SystemExit("no picks with gold masks in this run")
    scores = np.array([r["score"] for r in recs])
    ious = np.array([r["gold_iou"] for r in recs])
    accept, review = calibrate(scores, ious, args.target_mean_iou, target_good_frac=args.target_good_frac)
    try:
        from scipy.stats import spearmanr

        rho = float(spearmanr(scores, ious).statistic)
    except Exception:
        rho = float("nan")
    report = summarize(recs)
    sel_a, sel_r = scores >= accept, (scores >= review) & (scores < accept)
    report["spearman_score_vs_iou"] = round(rho, 4)
    report["recommended"] = {
        "accept_cutoff": accept, "review_cutoff": review,
        "accepted": {"n": int(sel_a.sum()), "mean_iou": round(float(ious[sel_a].mean()), 4) if sel_a.any() else None},
        "reviewed": {"n": int(sel_r.sum()), "mean_iou": round(float(ious[sel_r].mean()), 4) if sel_r.any() else None},
        "hard": {"n": int((scores < review).sum())},
    }
    out = Path(args.out)
    write_json(out / "autolabel_eval.json", report)
    calibration_figure(scores, ious, [r["source"] for r in recs], out / "calibration.png", accept, review)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
