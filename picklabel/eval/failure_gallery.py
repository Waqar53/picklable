"""Every false positive and false negative at the operating threshold, as crops you can tag.

Writes ``fp/`` and ``fn/`` crops, an ``index.html`` to browse them, and ``failures.csv`` with
an empty ``cause`` column. Tag causes by hand (``film``, ``occlusion``, ``tiny``,
``dark_on_dark``, ``lookalike``, ``duplicate``, ``label_error``); about 100 errors take an
hour. Run again and it prints the counts per cause, so the biggest bucket gets fixed first.

    picklabel gallery --preds report/eval/R3/preds.jsonl --gt data/gold --threshold 0.42 --out report/failures
"""

from __future__ import annotations

import argparse
import csv
import html
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from picklabel.eval.data import load_gt, load_predictions
from picklabel.eval.metrics import match_image
from picklabel.io import read_image
from picklabel.masks import iou_matrix, mask_bbox
from picklabel.viz import overlay

CAUSES = ["film", "occlusion", "tiny", "dark_on_dark", "lookalike", "duplicate", "label_error", "other"]


def crop(image: np.ndarray, mask: np.ndarray, color, pad: float = 0.6, size: int = 256) -> np.ndarray:
    x1, y1, x2, y2 = mask_bbox(mask) or (0, 0, image.shape[1], image.shape[0])
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad) + 8, int(h * pad) + 8
    X1, Y1 = max(0, x1 - px), max(0, y1 - py)
    X2, Y2 = min(image.shape[1], x2 + px), min(image.shape[0], y2 + py)
    out = overlay(image, mask, color, alpha=0.35)[Y1:Y2, X1:X2]
    s = size / max(out.shape[:2])
    return cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def collect(preds: list[dict], gts_by_image: dict, thr: float, iou_thr: float = 0.5) -> list[dict]:
    errors = []
    for rec in preds:
        gt = gts_by_image.get(rec["image"], [])
        keep = [i for i, s in enumerate(rec["scores"]) if s >= thr]
        masks = [rec["masks"][i] for i in keep]
        scores = [rec["scores"][i] for i in keep]
        ev = match_image(masks, scores, gt, iou_thr)
        ious = iou_matrix(masks, gt) if masks and gt else np.zeros((len(masks), len(gt)))
        for i, (m, s) in enumerate(zip(masks, scores)):
            if not ev.tp[i]:
                best = float(ious[i].max()) if gt else 0.0
                errors.append({"kind": "fp", "image": rec["image"], "mask": m, "score": s, "best_iou": best,
                               "area": int(m.sum()), "duplicate": bool(ev.duplicate[i])})
        for j, g in enumerate(gt):
            if ev.gt_best_score[j] < 0:
                best = float(ious[:, j].max()) if masks else 0.0
                errors.append({"kind": "fn", "image": rec["image"], "mask": g, "score": None, "best_iou": best,
                               "area": int(g.sum()), "duplicate": False})
    return errors


def write_gallery(errors: list[dict], out: Path) -> Path:
    for kind in ("fp", "fn"):
        (out / kind).mkdir(parents=True, exist_ok=True)
    images: dict[str, np.ndarray] = {}
    rows, cells = [], {"fp": [], "fn": []}
    for n, e in enumerate(errors):
        img = images.setdefault(e["image"], read_image(e["image"]))
        color = (60, 60, 230) if e["kind"] == "fp" else (0, 190, 255)
        name = f"{e['kind']}/{n:04d}.jpg"
        cv2.imwrite(str(out / name), crop(img, e["mask"], color), [cv2.IMWRITE_JPEG_QUALITY, 88])
        cause = "duplicate" if e["duplicate"] else ""
        rows.append({"id": n, "kind": e["kind"], "crop": name, "image": e["image"],
                     "score": "" if e["score"] is None else round(e["score"], 3),
                     "best_iou": round(e["best_iou"], 3), "area": e["area"], "cause": cause})
        label = f"{e['kind'].upper()} #{n} score={rows[-1]['score']} iou={rows[-1]['best_iou']} area={e['area']}"
        cells[e["kind"]].append(f'<figure><img src="{name}" loading="lazy"><figcaption>{html.escape(label)}'
                                f"</figcaption></figure>")
    with open(out / "failures.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["id"])
        w.writeheader()
        w.writerows(rows)
    page = ["<!doctype html><meta charset='utf-8'><title>PickLabel failures</title>",
            "<style>body{font:14px system-ui;margin:16px}figure{display:inline-block;margin:6px;width:256px}"
            "img{width:256px;border-radius:4px}figcaption{font-size:12px;color:#52514e}</style>",
            f"<h1>Failures ({len(errors)})</h1><p>Tag causes in failures.csv: {', '.join(CAUSES)}</p>"]
    for kind, title in (("fn", "Missed plastic (false negatives)"), ("fp", "False alarms (false positives)")):
        page.append(f"<h2>{title}: {len(cells[kind])}</h2>" + "".join(cells[kind]))
    (out / "index.html").write_text("\n".join(page))
    return out / "index.html"


def cause_counts(csv_path: Path) -> dict[str, Counter]:
    counts: dict[str, Counter] = {"fp": Counter(), "fn": Counter()}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("cause"):
                counts[row["kind"]][row["cause"]] += 1
    return counts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Crops of every FP and FN, for tagging failure causes.")
    ap.add_argument("--preds", required=True, help="preds.jsonl from eval-student or sam3-baseline")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--out", default="report/failures")
    args = ap.parse_args(argv)
    out = Path(args.out)
    csv_path = out / "failures.csv"
    if csv_path.exists() and any(cause_counts(csv_path)[k] for k in ("fp", "fn")):
        for kind, c in cause_counts(csv_path).items():
            print(kind.upper(), dict(c.most_common()))
        return
    gts_by_image = {str(g.image_path): g.masks for g in load_gt(args.gt, args.image_dir)}
    errors = collect(list(load_predictions(args.preds)), gts_by_image, args.threshold)
    index = write_gallery(errors, out)
    print(f"{sum(e['kind'] == 'fp' for e in errors)} FP, {sum(e['kind'] == 'fn' for e in errors)} FN -> {index}")


if __name__ == "__main__":
    main()
