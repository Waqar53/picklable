"""Evaluate a YOLO-seg student with the same metrics as the SAM 3 baseline, and time it.

    picklabel eval-student --weights runs/train/R3/weights/best.pt --gt data/gold --out report/eval/R3
    picklabel eval-student --weights best.pt --gt datasets/R3/images/val --latency --imgsz 1024

Predictions are made at conf 0.01 so the whole PR curve is available; the operating point is
chosen afterwards (recall @ P >= 0.90). Latency is measured at batch 1 after warm-up, with a
device sync around the timed loop.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from picklabel.eval.data import GTImage, load_gt, save_predictions
from picklabel.eval.metrics import evaluate
from picklabel.eval.plots import pr_figure
from picklabel.io import read_image, write_json
from picklabel.masks import resize_mask


def predict_yolo(weights: str, images: list[Path], imgsz: int = 1024, conf: float = 0.01,
                 device: str | None = None) -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(weights)
    out = []
    for path in images:
        img = read_image(path)
        r = model.predict(img, imgsz=imgsz, conf=conf, retina_masks=True, verbose=False, device=device)[0]
        masks, scores, labels = [], [], []
        if r.masks is not None and len(r.masks):
            h, w = img.shape[:2]
            for m, s, c in zip(r.masks.data.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                mk = m > 0.5
                if mk.shape != (h, w):
                    mk = resize_mask(mk, (w, h))
                masks.append(mk)
                scores.append(float(s))
                labels.append(r.names[int(c)])
        out.append({"image": str(path), "masks": masks, "scores": scores, "labels": labels})
    return out


def _sync(device: str | None) -> None:
    try:
        import torch

        if torch.cuda.is_available() and (device is None or str(device) not in ("cpu", "mps")):
            torch.cuda.synchronize()
        elif str(device) == "mps":
            torch.mps.synchronize()
    except Exception:
        pass


def benchmark(weights: str, image_path: Path, imgsz: int = 1024, n: int = 50, warmup: int = 10,
              device: str | None = None, half: bool = False) -> dict:
    from ultralytics import YOLO

    model = YOLO(weights)
    img = read_image(image_path)
    for _ in range(warmup):
        model.predict(img, imgsz=imgsz, verbose=False, device=device, half=half)
    times = []
    for _ in range(n):
        _sync(device)
        t = time.perf_counter()
        model.predict(img, imgsz=imgsz, verbose=False, device=device, half=half)
        _sync(device)
        times.append((time.perf_counter() - t) * 1000)
    t = np.array(times)
    return {"device": str(device), "imgsz": imgsz, "half": half, "n": n, "ms_mean": round(float(t.mean()), 2),
            "ms_p50": round(float(np.median(t)), 2), "ms_p90": round(float(np.percentile(t, 90)), 2),
            "fps": round(float(1000 / t.mean()), 1)}


def evaluate_records(records: list[dict], gts: list[GTImage], operating_thr: float | None = None):
    by_image = {str(g.image_path): g for g in gts}
    preds, gt_masks = [], []
    for rec in records:
        g = by_image[rec["image"]]
        preds.append((rec["masks"], rec["scores"]))
        gt_masks.append(g.masks)
    return evaluate(preds, gt_masks, operating_thr=operating_thr)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate a YOLO-seg student (and time it).")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gt", required=True, help="pairs folder with gt_instances, COCO json, or images/<split>")
    ap.add_argument("--image-dir", default=None, help="image folder for a COCO --gt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="student")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument("--threshold", type=float, default=None, help="fixed operating threshold (default: P>=0.90)")
    ap.add_argument("--latency", action="store_true")
    args = ap.parse_args(argv)

    gts = load_gt(args.gt, args.image_dir)
    if not gts:
        raise SystemExit(f"no ground truth found in {args.gt}")
    records = predict_yolo(args.weights, [g.image_path for g in gts], args.imgsz, device=args.device)
    out = Path(args.out)
    save_predictions(out / "preds.jsonl", records)
    report, curve = evaluate_records(records, gts, args.threshold)
    result = {"name": args.name, "weights": args.weights, "gt": args.gt, "imgsz": args.imgsz, **report.as_dict()}
    if args.latency:
        result["latency"] = benchmark(args.weights, gts[0].image_path, args.imgsz, device=args.device)
    write_json(out / "report.json", result)
    thr = report.at_threshold["threshold"]
    pr_figure({args.name: curve}, out / "pr.png", points={args.name: (thr, f"thr {thr:.2f}")})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
