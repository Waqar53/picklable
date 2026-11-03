"""SAM 3 zero-shot baseline: which text prompts, and which threshold?

SAM 3 runs exactly the way Grip's ``sam3_ros`` node runs it: Ultralytics
``SAM3SemanticPredictor``, ``set_image`` once, then one call per prompt. Every candidate is
cached once at a low confidence floor; everything after that is a sweep over the cache:

* prompt sets (``configs/prompt_sets.yaml``);
* ``node`` mode, the node's behaviour: every prompt's masks are published as they are, so one
  bag matched by "plastic bag" and by "plastic film" is two detections, i.e. two picks.
  ``nms`` mode adds cross-prompt mask NMS;
* the threshold: the node's default 0.75 against the best threshold for P >= 0.90, which can
  be set live on the node with ``ros2 param set /sam3_node threshold <t>``.

    picklabel sam3-baseline --weights weights/sam3.pt --gt data/gold --out report/sam3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from picklabel.eval.data import load_gt, load_predictions, save_predictions
from picklabel.eval.metrics import evaluate
from picklabel.eval.plots import pr_figure
from picklabel.io import read_image, write_json
from picklabel.masks import mask_nms

NODE_DEFAULT_THRESHOLD = 0.75
PROMPT_SETS_PATH = Path(__file__).resolve().parents[2] / "configs" / "prompt_sets.yaml"


def run_candidates(weights: str, images: list[Path], prompts: list[str], conf_floor: float = 0.05,
                   device: str | None = None) -> list[dict]:
    from picklabel.autolabel.segment import Sam3TextSegmenter, _results_to_candidates

    seg = Sam3TextSegmenter(weights, conf=conf_floor, device=device)
    records = []
    for path in images:
        img = read_image(path)
        t = time.perf_counter()
        seg.predictor.set_image(img)
        set_ms = (time.perf_counter() - t) * 1000
        masks, scores, labels, prompt_ms = [], [], [], {}
        for p in prompts:
            t = time.perf_counter()
            cands = _results_to_candidates(seg.predictor(text=[p]), img.shape[:2], "sam3", p)
            prompt_ms[p] = (time.perf_counter() - t) * 1000
            for c in cands:
                masks.append(c.mask)
                scores.append(c.score)
                labels.append(p)
        records.append({"image": str(path), "masks": masks, "scores": scores, "labels": labels,
                        "set_image_ms": round(set_ms, 1), "prompt_ms": {k: round(v, 1) for k, v in prompt_ms.items()}})
    return records


def assemble(record: dict, prompt_set: list[str], nms_iou: float | None) -> tuple[list[np.ndarray], list[float]]:
    idx = [i for i, lab in enumerate(record["labels"]) if lab in prompt_set]
    masks = [record["masks"][i] for i in idx]
    scores = [record["scores"][i] for i in idx]
    if nms_iou is not None and masks:
        keep = mask_nms(masks, scores, nms_iou)
        masks, scores = [masks[i] for i in keep], [scores[i] for i in keep]
    return masks, scores


def sweep(records: list[dict], gts_by_image: dict, prompt_sets: dict[str, list[str]], nms_iou: float = 0.6
          ) -> tuple[list[dict], dict]:
    rows, curves = [], {}
    for set_name, prompts in prompt_sets.items():
        for mode, iou in (("node", None), ("nms", nms_iou)):
            preds = [assemble(r, prompts, iou) for r in records]
            gts = [gts_by_image[r["image"]] for r in records]
            best, curve = evaluate(preds, gts)
            at_default, _ = evaluate(preds, gts, operating_thr=NODE_DEFAULT_THRESHOLD)
            lat = [r["set_image_ms"] + sum(r["prompt_ms"][p] for p in prompts) for r in records]
            rows.append({
                "prompt_set": set_name, "prompts": prompts, "mode": mode,
                "ap50": round(best.ap50, 4), "recall_at_p90": round(best.recall_at_p90, 4),
                "thr_at_p90": best.thr_at_p90,
                "at_0.75": at_default.at_threshold,
                "at_best": best.at_threshold,
                "ms_per_frame_p50": round(float(np.median(lat)), 1),
            })
            curves[f"{set_name}/{mode}"] = curve
    return rows, curves


def table_md(rows: list[dict]) -> str:
    lines = [
        "| prompts | mode | R @ P>=0.90 | thr for P>=0.90 | P / R at node default 0.75 | duplicates at 0.75 | AP50 | ms/frame |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        d = r["at_0.75"]
        thr = f"{r['thr_at_p90']:.2f}" if r["thr_at_p90"] is not None else "not reached"
        lines.append(f"| {r['prompt_set']}: {', '.join(r['prompts'])} | {r['mode']} | {r['recall_at_p90']:.3f} | {thr} | "
                     f"{d['precision']:.3f} / {d['recall']:.3f} | {d['duplicates']} | {r['ap50']:.3f} | "
                     f"{r['ms_per_frame_p50']:.0f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="SAM 3 prompt-set and threshold sweep (sam3_ros-compatible).")
    ap.add_argument("--weights", default="weights/sam3.pt")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--classes", default=None, help="comma list of GT classes to keep (e.g. soft_plastic,rigid_plastic)")
    ap.add_argument("--prompt-sets", default=str(PROMPT_SETS_PATH))
    ap.add_argument("--out", default="report/sam3")
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf-floor", type=float, default=0.05)
    ap.add_argument("--nms-iou", type=float, default=0.6)
    args = ap.parse_args(argv)

    with open(args.prompt_sets) as f:
        prompt_sets = yaml.safe_load(f)
    classes = args.classes.split(",") if args.classes else None
    gts = load_gt(args.gt, args.image_dir, classes)
    out = Path(args.out)
    cache = out / "sam3_candidates.jsonl"
    if cache.exists():
        records = list(load_predictions(cache))
    else:
        prompts = sorted({p for ps in prompt_sets.values() for p in ps})
        records = run_candidates(args.weights, [g.image_path for g in gts], prompts, args.conf_floor, args.device)
        save_predictions(cache, records)
    gts_by_image = {str(g.image_path): g.masks for g in gts}
    rows, curves = sweep(records, gts_by_image, prompt_sets, args.nms_iou)
    write_json(out / "report.json", {"gt": args.gt, "n_images": len(records), "rows": rows})
    (out / "table.md").write_text(table_md(rows) + "\n")
    best = max(rows, key=lambda r: r["recall_at_p90"])
    key = f"{best['prompt_set']}/{best['mode']}"
    node_key = f"{best['prompt_set']}/node"
    pr_figure({f"SAM 3 {key}": curves[key], f"SAM 3 {node_key}": curves[node_key]} if key != node_key
              else {f"SAM 3 {key}": curves[key]}, out / "pr_sam3.png",
              title="SAM 3 zero-shot: best prompt set",
              points={f"SAM 3 {node_key}": (NODE_DEFAULT_THRESHOLD, "node default 0.75")})
    print(table_md(rows))
    print(json.dumps({"best": best}, indent=2))


if __name__ == "__main__":
    main()
