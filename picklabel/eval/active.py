"""Active learning: which unlabelled frames should a human look at next?

The student runs on unlabelled frames (e.g. everything the robot saw today). Each frame gets
an uncertainty: how much of its detection mass sits near the operating threshold, where a
small score change flips a pick decision. The top-k frames go to Roboflow's review queue
with the student's masks attached as predictions, so a reviewer corrects them instead of
drawing from scratch.

    picklabel select-uncertain --weights best.pt --images data/unlabelled --threshold 0.42 --k 100 --push
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from picklabel.io import IMAGE_EXTS, write_json
from picklabel.roboflow_sync.coco import write_image_coco
from picklabel.roboflow_sync.upload import RoboflowSync, UploadItem, default_batch_name
from picklabel.types import Route


def frame_uncertainty(scores: list[float], threshold: float, band: float = 0.2) -> float:
    """Sum over detections of a triangle kernel centred on the threshold (1 at thr, 0 at +-band)."""
    s = np.asarray(scores, dtype=np.float64)
    return float(np.clip(1.0 - np.abs(s - threshold) / band, 0.0, 1.0).sum()) if s.size else 0.0


def rank_frames(records: list[dict], threshold: float, band: float = 0.2) -> list[dict]:
    ranked = [{"image": r["image"], "uncertainty": round(frame_uncertainty(r["scores"], threshold, band), 4),
               "n_near_threshold": int(np.sum(np.abs(np.asarray(r["scores"]) - threshold) < band)),
               "record": r} for r in records]
    return sorted(ranked, key=lambda x: -x["uncertainty"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Pick the most uncertain unlabelled frames for review.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True, help="folder of unlabelled frames")
    ap.add_argument("--threshold", type=float, required=True, help="the student's operating threshold")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--band", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/active")
    ap.add_argument("--push", action="store_true", help="upload the selection to Roboflow's review queue")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from picklabel.eval.eval_student import predict_yolo

    images = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    records = predict_yolo(args.weights, images, args.imgsz, conf=max(args.threshold - args.band, 0.01),
                           device=args.device)
    ranked = [r for r in rank_frames(records, args.threshold, args.band) if r["uncertainty"] > 0][: args.k]
    out = Path(args.out)
    sync = RoboflowSync(dry_run=args.dry_run or not args.push, ledger_path=out / "roboflow_ledger.jsonl") \
        if (args.push or args.dry_run) else None
    for i, item in enumerate(ranked):
        rec = item.pop("record")
        if sync is None:
            continue
        img = Path(rec["image"])
        h, w = rec["masks"][0].shape  # every ranked frame has at least one detection near the threshold
        coco = write_image_coco(out / "coco" / f"{img.stem}.json", img.name, h, w,
                                [(m, lab or "plastic", {"route": "review", "score": round(s, 4), "source": "student"})
                                 for m, s, lab in zip(rec["masks"], rec["scores"], rec["labels"])], ["plastic"])
        sync.push(UploadItem(key=f"active__{img.stem}", image_path=img, route=Route.REVIEW, coco_path=coco,
                             n_instances=len(rec["masks"]),
                             tags=["picklabel", "active-learning", "route:review", f"rank:{i}"],
                             metadata={"uncertainty": item["uncertainty"], "threshold": args.threshold}),
                  default_batch_name("active"))
    write_json(out / "selection.json", ranked)
    print(json.dumps(ranked[:10], indent=2))


if __name__ == "__main__":
    main()
