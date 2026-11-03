"""Batch auto-labeling: a folder of pick pairs in; images, COCO, previews, a manifest and a summary out.

    picklabel autolabel data/pairs --out runs/autolabel/home-day1 --sam2 weights/sam2.1_t.pt
    picklabel autolabel data/pairs --out runs/autolabel/home-day1 --sam3 weights/sam3.pt --push

Outputs::

    <out>/images/<session>__<pick>.jpg   the before frame (the image that gets labelled)
    <out>/coco/<session>__<pick>.json    single-image COCO, as uploaded to Roboflow
    <out>/annotations.json               dataset COCO (accepted + review instances, negatives)
    <out>/previews/<session>__<pick>.jpg before | after | change | mask, score, route
    <out>/manifest.jsonl                 one record per pick (route, score, signals, timings, gold IoU)
    <out>/summary.json                   route counts, yield, gold IoU by route, timings
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from picklabel.autolabel.pipeline import AutoLabelConfig, AutoLabeler, is_failed_pick
from picklabel.autolabel.propagate import FrameLabels, propagate_session, single_frames
from picklabel.io import find_pick_dirs, load_pair, load_session_meta, read_image, write_image, write_json
from picklabel.masks import instances_from_id_map, iou_matrix, mask_iou
from picklabel.roboflow_sync.coco import CocoBuilder, write_image_coco
from picklabel.roboflow_sync.upload import RoboflowSync, UploadItem, default_batch_name, make_tags
from picklabel.types import AutoLabel, PickPair
from picklabel.viz import autolabel_panel


def build_labeler(cfg: AutoLabelConfig, sam3: str | None = None, sam2: str | None = None,
                  device: str | None = None) -> AutoLabeler:
    text_seg = point_seg = None
    if sam3:
        from picklabel.autolabel.segment import Sam3TextSegmenter

        text_seg = Sam3TextSegmenter(sam3, conf=cfg.segment.sam3_conf, device=device)
    if sam2:
        from picklabel.autolabel.segment import Sam2PointSegmenter

        point_seg = Sam2PointSegmenter(sam2, device=device)
    return AutoLabeler(cfg, text_segmenter=text_seg, point_segmenter=point_seg)


def _frame_match(frame: FrameLabels, gt_ids: np.ndarray) -> tuple[float, float]:
    """Instance recall and precision of a frame's labels against its full ground truth (IoU >= 0.5)."""
    gts = instances_from_id_map(gt_ids)
    preds = [i.mask for i in frame.instances]
    if not gts:
        return 1.0, 1.0 if not preds else 0.0
    if not preds:
        return 0.0, 1.0
    ious = iou_matrix(preds, gts)
    matched_gt, tp = set(), 0
    for p in np.argsort(-ious.max(axis=1)):
        g = int(np.argmax(ious[p]))
        if ious[p, g] >= 0.5 and g not in matched_gt:
            matched_gt.add(g)
            tp += 1
    return tp / len(gts), tp / len(preds)


def _summarise(records: list[dict], n_negatives: int, elapsed_s: float) -> dict:
    routes = Counter(r["route"] for r in records)
    n = len(records)
    summary: dict = {
        "n_picks": n,
        "n_negative_frames": n_negatives,
        "routes": {k: routes.get(k, 0) for k in ("accept", "review", "hard")},
        "label_yield": round((routes.get("accept", 0) + routes.get("review", 0)) / n, 4) if n else 0.0,
        "sources": dict(Counter(r["source"] for r in records)),
        "mean_quality": round(float(np.mean([r["score"] for r in records])), 4) if n else 0.0,
        "elapsed_s": round(elapsed_s, 1),
        "ms_per_pick_median": {},
    }
    stages = defaultdict(list)
    for r in records:
        for k, v in r["timings_ms"].items():
            stages[k].append(v)
    summary["ms_per_pick_median"] = {k: round(float(np.median(v)), 1) for k, v in stages.items()}

    gold = [r for r in records if r.get("gold_iou") is not None]
    if gold:
        by_route = {}
        for route in ("accept", "review", "hard"):
            ious = [r["gold_iou"] for r in gold if r["route"] == route and not r.get("failed_pick")]
            if ious:
                by_route[route] = {
                    "n": len(ious), "mean_iou": round(float(np.mean(ious)), 4),
                    "iou>=0.75": round(float(np.mean([i >= 0.75 for i in ious])), 4),
                }
        labelled = [r["gold_iou"] for r in gold if r["route"] in ("accept", "review")]
        summary["gold"] = {
            "n": len(gold),
            "mean_iou_labelled": round(float(np.mean(labelled)), 4) if labelled else None,
            "by_route": by_route,
        }
        truly_missed = [r for r in gold if r.get("truly_missed")]
        if truly_missed:
            caught = sum(1 for r in truly_missed if r.get("failed_pick"))
            summary["gold"]["missed_picks_caught"] = f"{caught}/{len(truly_missed)}"
        fr = [r for r in gold if r.get("frame_recall") is not None]
        if fr:
            summary["gold"]["frame_instance_recall"] = round(float(np.mean([r["frame_recall"] for r in fr])), 4)
            summary["gold"]["frame_instance_precision"] = round(float(np.mean([r["frame_precision"] for r in fr])), 4)
            summary["gold"]["frames_complete"] = round(float(np.mean([r["frame_complete"] for r in fr])), 4)
    return summary


def run(pairs_root: str | Path, out_dir: str | Path, labeler: AutoLabeler, previews: bool = True,
        limit: int | None = None, sync: RoboflowSync | None = None, batch_name: str | None = None,
        origin: str = "home", verbose: bool = True) -> dict:
    cfg = labeler.cfg
    out = Path(out_dir)
    for sub in ("images", "coco") + (("previews",) if previews else ()):
        (out / sub).mkdir(parents=True, exist_ok=True)
    dirs = find_pick_dirs(pairs_root)[:limit] if limit else find_pick_dirs(pairs_root)
    sessions: dict[Path, list[Path]] = defaultdict(list)
    for d in dirs:
        sessions[d.parent].append(d)

    coco = CocoBuilder(["plastic"])
    records: list[dict] = []
    n_negatives = 0
    batch_name = batch_name or default_batch_name(origin)
    t0 = time.time()
    manifest = open(out / "manifest.jsonl", "w")
    try:
        for session_dir, pick_dirs in sessions.items():
            pairs: list[PickPair] = []
            for d in pick_dirs:
                try:
                    pairs.append(load_pair(d))
                except ValueError as e:  # e.g. imported photos not annotated yet
                    print(f"skip {d}: {e}")
            if not pairs:
                continue
            labels: list[AutoLabel] = [labeler.label(p, keep_debug=previews) for p in pairs]
            meta = load_session_meta(session_dir)
            cleared = bool(meta.get("cleared", False))
            if cfg.propagate.enabled and cfg.align.mode == "static" and len(pairs) > 1:
                frames = propagate_session(pairs, labels, cfg.propagate, cfg.align, cfg.change, cleared)
            else:
                frames = single_frames(labels)
            for pair, label, frame in zip(pairs, labels, frames):
                rec = _write_frame(out, coco, pair, label, frame, previews, sync, batch_name)
                manifest.write(json.dumps(rec) + "\n")
                records.append(rec)
                if verbose:
                    gold = f" gold IoU {rec['gold_iou']:.2f}" if rec.get("gold_iou") is not None else ""
                    print(f"[{label.session}/{label.pick_id}] {label.route.value:6s} q={label.score:.2f} "
                          f"{label.source:6s} inst={len(frame.instances)}{gold} {' '.join(label.reasons)}")
            final = meta.get("final_after")
            if cleared and final and (session_dir / final).exists():
                n_negatives += 1
                name = f"{session_dir.name}__final_after.jpg"
                img = read_image(session_dir / final)
                write_image(out / "images" / name, img)
                coco.add_image(name, img.shape[0], img.shape[1], {"negative": True, "session": session_dir.name})
    finally:
        manifest.close()

    coco.save(out / "annotations.json")
    summary = _summarise(records, n_negatives, time.time() - t0)
    summary["config"] = cfg.to_dict()
    summary["models"] = {
        "text_segmenter": type(labeler.text_segmenter).__name__ if labeler.text_segmenter else None,
        "point_segmenter": type(labeler.point_segmenter).__name__ if labeler.point_segmenter else None,
    }
    write_json(out / "summary.json", summary)
    return summary


def _write_frame(out: Path, coco: CocoBuilder, pair: PickPair, label: AutoLabel, frame: FrameLabels,
                 previews: bool, sync: RoboflowSync | None, batch_name: str) -> dict:
    name = f"{pair.session}__{pair.pick_id}"
    h, w = pair.shape
    image_path = write_image(out / "images" / f"{name}.jpg", pair.before)
    instances = [
        (inst.mask, pair.class_name, {
            "route": inst.route.value, "score": round(inst.score, 4), "source": inst.source,
            "from_pick": inst.from_pick, "propagated": inst.propagated,
        })
        for inst in frame.instances
    ]
    coco_path = write_image_coco(out / "coco" / f"{name}.json", image_path.name, h, w, instances, ["plastic"])
    image_id = coco.add_image(image_path.name, h, w, {
        "session": pair.session, "pick_id": pair.pick_id, "route": frame.route.value,
        "complete": frame.complete, "grasp_px": list(label.grasp_xy) if label.grasp_xy else None,
    })
    for mask, cls, extra in instances:
        coco.add_instance(image_id, mask, cls, extra)

    rec = label.to_record()
    rec.update(
        image=image_path.name,
        frame_route=frame.route.value,
        frame_complete=frame.complete,
        n_instances=len(frame.instances),
        notes=frame.notes,
        failed_pick=is_failed_pick(label),
    )
    if pair.meta.get("missed") is not None:
        rec["truly_missed"] = bool(pair.meta["missed"])
    if pair.gt_mask is not None:
        rec["gold_iou"] = round(mask_iou(label.mask, pair.gt_mask), 4) if label.mask is not None else 0.0
    if pair.gt_instances is not None:
        recall, precision = _frame_match(frame, pair.gt_instances)
        rec["frame_recall"], rec["frame_precision"] = round(recall, 4), round(precision, 4)

    if previews:
        cv2.imwrite(str(out / "previews" / f"{name}.jpg"), autolabel_panel(pair, label),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    if sync is not None:
        item = UploadItem(
            key=name, image_path=image_path, route=frame.route, coco_path=coco_path,
            n_instances=len(frame.instances),
            tags=make_tags(frame.route, label.source, label.score, pair.session, label.reasons,
                           ["complete" if frame.complete else "incomplete"]),
            metadata={"pick_id": pair.pick_id, "session": pair.session, "quality": round(label.score, 3),
                      "route": frame.route.value, "source": label.source, "n_instances": len(frame.instances),
                      **{k: round(float(v), 3) for k, v in label.signals.items()
                         if k in ("agreement", "support", "contrast", "shape")}},
        )
        rec["roboflow"] = sync.push(item, batch_name)
    return rec


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Auto-label pick pairs and (optionally) push them to Roboflow.")
    ap.add_argument("pairs", help="folder of pick pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="YAML config (default: built-in, same as configs/autolabel.yaml)")
    ap.add_argument("--sam3", default=None, help="path to sam3.pt: text-prompted candidates")
    ap.add_argument("--sam2", default=None, help="SAM 2.1 weights, e.g. sam2.1_t.pt: grasp-prompted masks")
    ap.add_argument("--device", default=None, help="cpu | mps | cuda | cuda:0")
    ap.add_argument("--mode", choices=["static", "belt"], default=None, help="override align.mode")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-previews", action="store_true")
    ap.add_argument("--push", action="store_true", help="upload to Roboflow (needs .env)")
    ap.add_argument("--dry-run", action="store_true", help="record the Roboflow calls without sending them")
    ap.add_argument("--batch", default=None, help="Roboflow batch name")
    ap.add_argument("--origin", default="home", help="data origin used in the batch name: home | robot | synthetic")
    args = ap.parse_args(argv)

    cfg = AutoLabelConfig.from_yaml(args.config) if args.config else AutoLabelConfig()
    if args.mode:
        cfg.align.mode = args.mode
    labeler = build_labeler(cfg, args.sam3, args.sam2, args.device)
    sync = None
    if args.push or args.dry_run:
        sync = RoboflowSync(dry_run=args.dry_run or not args.push,
                            ledger_path=Path(args.out) / "roboflow_ledger.jsonl")
    summary = run(args.pairs, args.out, labeler, previews=not args.no_previews, limit=args.limit, sync=sync,
                  batch_name=args.batch, origin=args.origin)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
