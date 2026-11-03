"""Summarise the WS2 real-image benchmark runs into report/ws2/ (tables, calibration, example strip).

    python scripts/ws2_report.py --run change=runs/ws2_change --run sam2=runs/ws2_sam2 --out report/ws2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from picklabel.eval.autolabel_eval import calibrate  # noqa: E402
from picklabel.eval.plots import calibration_figure  # noqa: E402
from picklabel.io import write_json  # noqa: E402

NAMES = {"change": "change map + GrabCut", "sam2": "SAM 2.1-tiny at the grasp"}


def load(run: Path) -> list[dict]:
    with open(run / "manifest.jsonl") as f:
        return [json.loads(line) for line in f]


def summarise(recs: list[dict]) -> dict:
    removed = [r for r in recs if not r.get("truly_missed")]
    missed = [r for r in recs if r.get("truly_missed")]
    iou = np.array([r["gold_iou"] for r in removed])
    score = np.array([r["score"] for r in removed])
    out = {
        "n_picks": len(recs), "n_removed": len(removed), "n_missed": len(missed),
        "mean_iou": round(float(iou.mean()), 4), "iou>=0.75": round(float((iou >= 0.75).mean()), 4),
        "iou>=0.5": round(float((iou >= 0.5).mean()), 4),
        "missed_caught": f"{sum(r['failed_pick'] for r in missed)}/{len(missed)}",
        "routes": {k: sum(r["route"] == k for r in recs) for k in ("accept", "review", "hard")},
        "ms_per_pick_median": round(float(np.median([r["timings_ms"].get("total", 0) for r in recs])), 1),
    }
    for route in ("accept", "review", "hard"):
        v = np.array([r["gold_iou"] for r in removed if r["route"] == route])
        if v.size:
            out[f"{route}_mean_iou"] = round(float(v.mean()), 4)
    try:
        from scipy.stats import spearmanr

        out["spearman_score_iou"] = round(float(spearmanr(score, iou).statistic), 4)
    except Exception:
        pass
    # Training-data bar: 95% of auto-accepted labels at IoU >= 0.75; the rest goes to human review.
    acc, rev = calibrate(score, iou, target_mean_iou=0.85, target_good_frac=0.95)
    sel, sel_r = score >= acc, (score >= rev) & (score < acc)
    out["calibrated"] = {"target": "95% of accepted labels at IoU >= 0.75", "accept_cutoff": acc,
                         "review_cutoff": rev, "accepted_n": int(sel.sum()),
                         "accepted_mean_iou": round(float(iou[sel].mean()), 4) if sel.any() else None,
                         "accepted_iou>=0.75": round(float((iou[sel] >= 0.75).mean()), 4) if sel.any() else None,
                         "review_n": int(sel_r.sum()),
                         "review_mean_iou": round(float(iou[sel_r].mean()), 4) if sel_r.any() else None}
    return out


def strip(run: Path, recs: list[dict], out_path: Path, width: int = 1500) -> list[str]:
    """Best three real removals, one caught missed grasp, the worst removal: what works and what doesn't."""
    removed = sorted([r for r in recs if not r.get("truly_missed")], key=lambda r: -r["gold_iou"])
    missed = [r for r in recs if r.get("truly_missed") and r["failed_pick"]]
    chosen = removed[:3] + missed[:1] + removed[-1:]
    tiles = []
    for r in chosen:
        img = cv2.imread(str(run / "previews" / r["image"]))
        if img is None:
            continue
        s = width / img.shape[1]
        tiles.append(cv2.resize(img, (width, round(img.shape[0] * s)), interpolation=cv2.INTER_AREA))
        tiles.append(np.full((8, width, 3), 255, np.uint8))
    cv2.imwrite(str(out_path), np.vstack(tiles[:-1]), [cv2.IMWRITE_JPEG_QUALITY, 86])
    return [f"{r['image']} route={r['route']} iou={r['gold_iou']}" for r in chosen]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=run_dir")
    ap.add_argument("--out", default="report/ws2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary, rows = {}, []
    for spec in args.run:
        name, run = spec.split("=", 1)
        recs = load(Path(run))
        s = summarise(recs)
        summary[name] = s
        removed = [r for r in recs if not r.get("truly_missed")]
        calibration_figure([r["score"] for r in removed], [r["gold_iou"] for r in removed],
                           [NAMES.get(name, name)] * len(removed), out / f"calibration_{name}.png",
                           s["calibrated"]["accept_cutoff"], s["calibrated"]["review_cutoff"])
        s["example_strip"] = strip(Path(run), recs, out / f"examples_{name}.jpg")
        c = s["calibrated"]
        rows.append(f"| {NAMES.get(name, name)} | {s['mean_iou']:.3f} | {s['iou>=0.75']:.0%} | {s['iou>=0.5']:.0%} | "
                    f"{s['missed_caught']} | {c['accepted_n']}/{s['n_removed']} at IoU {c['accepted_mean_iou']:.3f} | "
                    f"{c['review_n']} at IoU {c['review_mean_iou'] or 0:.2f} | "
                    f"{s.get('spearman_score_iou', float('nan')):.2f} | {s['ms_per_pick_median'] / 1000:.1f} s |")
    write_json(out / "summary.json", summary)
    table = ("| Mask source | mean IoU | IoU >= 0.75 | IoU >= 0.5 | missed grasps caught | auto-accepted | sent to review "
             "| score vs IoU (Spearman) | time / pick |\n|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))
    (out / "table.md").write_text(table + "\n")
    print(table)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
