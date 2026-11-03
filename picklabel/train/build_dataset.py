"""Build a YOLO-seg dataset from auto-label runs (plus, optionally, extra COCO sources).

* Splits by capture session, never by image. Frames of one session are near-duplicates,
  so an image-level split leaks test scenes into training and inflates every number.
* Keeps a frame only if every instance on it has an allowed route, and (unless
  ``--allow-incomplete``) only if its labels are complete. A frame with one unlabelled
  plastic teaches the student that plastic is background.
* Adds cleared-tray frames as negatives: organics and look-alikes, no plastic.

    picklabel build-dataset runs/autolabel/home-* --out datasets/R2 --routes accept
    picklabel build-dataset runs/autolabel/home-* --out datasets/R3 --routes accept,review
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from picklabel.io import read_json, write_json


def _merge_segments(polys: list[list[float]]) -> np.ndarray:
    """One polygon per instance, as YOLO-seg needs. Multi-part masks are stitched (Ultralytics' method)."""
    segs = [np.asarray(p, np.float64).reshape(-1, 2) for p in polys if len(p) >= 6]
    if len(segs) == 1:
        return segs[0]
    try:
        from ultralytics.data.converter import merge_multi_segment

        return np.concatenate(merge_multi_segment([s.reshape(-1).tolist() for s in segs]), axis=0)
    except Exception:
        return max(segs, key=lambda s: len(s))


def yolo_lines(anns: list[dict], cls_index: dict[str, int], cat_names: dict[int, str], w: int, h: int) -> list[str]:
    lines = []
    for a in anns:
        name = cat_names.get(a["category_id"])
        if name not in cls_index or not isinstance(a.get("segmentation"), list) or not a["segmentation"]:
            continue
        seg = _merge_segments(a["segmentation"])
        if len(seg) < 3:
            continue
        xy = np.clip(seg / [w, h], 0.0, 1.0).reshape(-1)
        lines.append(f"{cls_index[name]} " + " ".join(f"{v:.6f}" for v in xy))
    return lines


def _split_sessions(sessions: list[str], val_frac: float, seed: int, pinned_test: set[str]) -> dict[str, str]:
    free = sorted(s for s in sessions if s not in pinned_test)
    rng = np.random.default_rng(seed)
    rng.shuffle(free)
    n_val = int(round(val_frac * len(free)))
    if len(free) >= 2:
        n_val = min(max(n_val, 1), len(free) - 1)
    split = {s: "val" for s in free[:n_val]}
    split.update({s: "train" for s in free[n_val:]})
    split.update({s: "test" for s in pinned_test})
    return split


def export_gt_run(pairs_root: str | Path, out_dir: str | Path) -> Path:
    """Write a pairs folder's ground truth (``gt_instances.png``) as if it were an auto-label run.

    Gives the manual-label reference (R1) the same layout as PickLabel runs, so both go
    through ``build`` and the same split.
    """
    from picklabel.io import find_pick_dirs, load_pair, load_session_meta, read_image, write_image
    from picklabel.masks import instances_from_id_map
    from picklabel.roboflow_sync.coco import CocoBuilder

    out = Path(out_dir)
    coco = CocoBuilder(["plastic"])
    sessions = set()
    for d in find_pick_dirs(pairs_root):
        pair = load_pair(d)
        if pair.gt_instances is None:
            continue
        name = f"{pair.session}__{pair.pick_id}.jpg"
        write_image(out / "images" / name, pair.before)
        image_id = coco.add_image(name, *pair.shape, {"session": pair.session, "pick_id": pair.pick_id,
                                                       "route": "accept", "complete": True})
        for m in instances_from_id_map(pair.gt_instances):
            coco.add_instance(image_id, m, pair.class_name, {"route": "accept", "source": "manual"})
        sessions.add(d.parent)
    for s in sorted(sessions):
        meta = load_session_meta(s)
        final = meta.get("final_after")
        if meta.get("cleared") and final and (s / final).exists():
            name = f"{s.name}__final_after.jpg"
            img = read_image(s / final)
            write_image(out / "images" / name, img)
            coco.add_image(name, img.shape[0], img.shape[1], {"negative": True, "session": s.name})
    coco.save(out / "annotations.json")
    return out


def build(runs: list[str | Path], out: str | Path, routes: set[str] | None = None, val_frac: float = 0.2,
          seed: int = 0, allow_incomplete: bool = False, negatives: bool = True,
          test_sessions: set[str] | None = None, names: list[str] | None = None,
          extra_coco: list[tuple[Path, Path, dict[str, str]]] | None = None) -> dict:
    routes = routes or {"accept"}
    names = names or ["plastic"]
    cls_index = {n: i for i, n in enumerate(names)}
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    frames = []  # (session, src_image, w, h, anns, cat_names)
    skipped = Counter()
    for run in runs:
        run = Path(run)
        coco = read_json(run / "annotations.json")
        cat_names = {c["id"]: c["name"] for c in coco["categories"]}
        anns_by_img = defaultdict(list)
        for a in coco["annotations"]:
            anns_by_img[a["image_id"]].append(a)
        for img in coco["images"]:
            meta = img.get("picklabel", {})
            anns = anns_by_img[img["id"]]
            if meta.get("negative"):
                if not negatives:
                    continue
            else:
                inst_routes = {a.get("picklabel", {}).get("route", "accept") for a in anns}
                if not anns:
                    skipped["no_labels"] += 1
                    continue
                if not inst_routes <= routes:
                    skipped["route"] += 1
                    continue
                if not meta.get("complete", False) and not allow_incomplete:
                    skipped["incomplete"] += 1
                    continue
            frames.append((meta.get("session", run.name), run / "images" / img["file_name"], img["width"],
                           img["height"], anns, cat_names))
    for coco_path, image_dir, class_map in extra_coco or []:
        coco = read_json(coco_path)
        cat_names = {c["id"]: class_map.get(c["name"], c["name"]) for c in coco["categories"]}
        anns_by_img = defaultdict(list)
        for a in coco["annotations"]:
            anns_by_img[a["image_id"]].append(a)
        for img in coco["images"]:
            frames.append((f"extra:{Path(coco_path).stem}", Path(image_dir) / img["file_name"], img["width"],
                           img["height"], anns_by_img[img["id"]], cat_names))

    sessions = sorted({f[0] for f in frames if not f[0].startswith("extra:")})
    split_of = _split_sessions(sessions, val_frac, seed, set(test_sessions or ()))
    counts = Counter()
    instances = Counter()
    for session, src, w, h, anns, cat_names in frames:
        split = "train" if session.startswith("extra:") else split_of[session]
        stem = f"{session.replace(':', '_')}__{src.stem}" if session.startswith("extra:") else src.stem
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / "images" / split / f"{stem}{src.suffix}")
        lines = yolo_lines(anns, cls_index, cat_names, w, h)
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        counts[split] += 1
        instances[split] += len(lines)

    out.mkdir(parents=True, exist_ok=True)
    if not counts:
        print(f"warning: no frames passed the filters (skipped: {dict(skipped)})")
    data = {"path": str(out.resolve()), "train": "images/train", "val": "images/val",
            "names": {i: n for i, n in enumerate(names)}}
    if counts.get("test"):
        data["test"] = "images/test"
    with open(out / "data.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    manifest = {
        "runs": [str(r) for r in runs], "routes": sorted(routes), "allow_incomplete": allow_incomplete,
        "images": dict(counts), "instances": dict(instances), "skipped": dict(skipped),
        "sessions": {s: split_of[s] for s in sessions}, "seed": seed,
    }
    write_json(out / "manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Auto-label runs -> YOLO-seg dataset with a session-level split.")
    ap.add_argument("runs", nargs="+", help="auto-label run folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--routes", default="accept", help="comma list: accept | accept,review")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--no-negatives", action="store_true")
    ap.add_argument("--test-sessions", default="", help="comma list of sessions pinned to the test split")
    ap.add_argument("--extra-coco", action="append", default=[],
                    help="COCO_JSON:IMAGE_DIR[:src=dst,...] added to train, e.g. zerowaste soft_plastic=plastic")
    args = ap.parse_args(argv)
    extra = []
    for spec in args.extra_coco:
        parts = spec.split(":")
        cmap = dict(kv.split("=") for kv in parts[2].split(",")) if len(parts) > 2 and parts[2] else {}
        extra.append((Path(parts[0]), Path(parts[1]), cmap))
    manifest = build(args.runs, args.out, set(args.routes.split(",")), args.val_frac, args.seed,
                     args.allow_incomplete, not args.no_negatives,
                     {s for s in args.test_sessions.split(",") if s}, extra_coco=extra)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
