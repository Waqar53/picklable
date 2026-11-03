"""Tiled inference for small fragments.

At full-frame resolution a 1-2 cm film fragment is a handful of pixels after the resize to
``imgsz``. Running the student on overlapping tiles (plus the full frame for large items)
and merging with mask NMS recovers them, at a known latency cost. ``main`` compares both
modes on small objects only, so the trade-off is measured rather than assumed.

    picklabel tiled --weights best.pt --gt data/gold --tile 640 --overlap 0.25
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from picklabel.eval.data import load_gt
from picklabel.eval.metrics import evaluate
from picklabel.io import read_image, write_json
from picklabel.masks import mask_nms, resize_mask


def tile_origins(length: int, tile: int, overlap: float) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    starts = list(range(0, length - tile, step))
    return starts + [length - tile]


def _predict(model, img: np.ndarray, imgsz: int, conf: float, max_det: int = 100
             ) -> tuple[list[np.ndarray], list[float]]:
    r = model.predict(img, imgsz=imgsz, conf=conf, retina_masks=True, max_det=max_det, verbose=False)[0]
    if r.masks is None:
        return [], []
    h, w = img.shape[:2]
    masks = [resize_mask(m > 0.5, (w, h)) for m in r.masks.data.cpu().numpy()]
    return masks, [float(s) for s in r.boxes.conf.cpu().numpy()]


def tiled_predict(model, img: np.ndarray, tile: int = 640, overlap: float = 0.25, imgsz: int = 640,
                  conf: float = 0.05, include_full: bool = True, nms_iou: float = 0.5, max_det: int = 50
                  ) -> tuple[list[np.ndarray], list[float]]:
    """``conf`` and ``max_det`` bound the masks per tile: NMS over full-frame masks is quadratic."""
    h, w = img.shape[:2]
    masks, scores = [], []
    if include_full:
        masks, scores = _predict(model, img, imgsz, conf, max_det)
    for y in tile_origins(h, tile, overlap):
        for x in tile_origins(w, tile, overlap):
            tm, ts = _predict(model, img[y:y + tile, x:x + tile], imgsz, conf, max_det)
            for m, s in zip(tm, ts):
                full = np.zeros((h, w), bool)
                full[y:y + m.shape[0], x:x + m.shape[1]] = m
                masks.append(full)
                scores.append(s)
    keep = mask_nms(masks, scores, nms_iou)
    return [masks[i] for i in keep], [scores[i] for i in keep]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Full-frame vs tiled inference on small objects.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--small-area", type=int, default=32 ** 2 * 4, help="GT area (px) below which an item is 'small'")
    ap.add_argument("--out", default="report/tiled")
    args = ap.parse_args(argv)
    from ultralytics import YOLO

    model = YOLO(args.weights)
    gts = load_gt(args.gt, args.image_dir)
    results = {}
    for mode in ("full", "tiled"):
        preds, small_gts, ms = [], [], []
        for g in gts:
            img = read_image(g.image_path)
            t = time.perf_counter()
            if mode == "full":
                p = _predict(model, img, args.imgsz, 0.05, 50)
            else:
                p = tiled_predict(model, img, args.tile, args.overlap, args.imgsz)
            ms.append((time.perf_counter() - t) * 1000)
            preds.append(p)
            small_gts.append([m for m in g.masks if m.sum() < args.small_area])
        rep, _ = evaluate(preds, small_gts)
        results[mode] = {"small_objects": rep.as_dict(), "ms_per_frame": round(float(np.median(ms)), 1)}
    write_json(Path(args.out) / "tiled.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
