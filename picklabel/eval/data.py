"""Ground truth and prediction I/O shared by every evaluation script.

Ground truth can come from:
* a PickLabel pairs folder whose picks carry ``gt_instances.png`` (gold set, synthetic):
  the image is each pick's before frame;
* a COCO JSON plus its image folder (ZeroWaste-f, a Roboflow COCO export);
* a YOLO-seg dataset split (``.../images/<split>`` with ``.../labels/<split>``).

Predictions are stored as JSON lines with COCO RLE masks, so a slow model (SAM 3) runs once
and every threshold sweep afterwards reads the cache.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from picklabel.io import IMAGE_EXTS, find_pick_dirs, read_id_map
from picklabel.masks import instances_from_id_map, polygons_to_mask, rle_decode, rle_encode


@dataclass
class GTImage:
    image_path: Path
    masks: list[np.ndarray]
    classes: list[str]


def _from_pairs(root: Path) -> list[GTImage]:
    out = []
    for d in find_pick_dirs(root):
        ids = d / "gt_instances.png"
        if not ids.exists():
            continue
        image = next(d / f"before{e}" for e in IMAGE_EXTS if (d / f"before{e}").exists())
        masks = instances_from_id_map(read_id_map(ids))
        out.append(GTImage(image, masks, ["plastic"] * len(masks)))
    return out


def _from_coco(path: Path, image_dir: Path | None, classes: Iterable[str] | None) -> list[GTImage]:
    from picklabel.roboflow_sync.coco import load_coco

    image_dir = image_dir or path.parent
    data = load_coco(path, classes)
    out = []
    for name, (_, _, insts) in data.items():
        p = image_dir / name
        if not p.exists():
            p = image_dir / "images" / name
        out.append(GTImage(p, [m for m, _, _ in insts], [c for _, c, _ in insts]))
    return out


def _from_yolo(images_dir: Path, names: list[str] | None) -> list[GTImage]:
    labels_dir = images_dir.parent.parent / "labels" / images_dir.name  # <root>/images/<split> -> labels/<split>
    out = []
    for img_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        h, w = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED).shape[:2]
        masks, classes = [], []
        txt = labels_dir / f"{img_path.stem}.txt"
        if txt.exists():
            for line in txt.read_text().splitlines():
                vals = line.split()
                if len(vals) < 7:
                    continue
                cls = int(vals[0])
                xy = np.array(vals[1:], np.float64).reshape(-1, 2) * [w, h]
                masks.append(polygons_to_mask([xy.reshape(-1).tolist()], h, w))
                classes.append(names[cls] if names and cls < len(names) else str(cls))
        out.append(GTImage(img_path, masks, classes))
    return out


def load_gt(source: str | Path, image_dir: str | Path | None = None, classes: Iterable[str] | None = None,
            names: list[str] | None = None) -> list[GTImage]:
    """Load ground truth from a pairs folder, a COCO JSON, or a YOLO ``images/<split>`` folder."""
    source = Path(source)
    if source.suffix == ".json":
        gts = _from_coco(source, Path(image_dir) if image_dir else None, classes)
    elif source.is_dir() and source.name in ("train", "val", "test") and source.parent.name == "images":
        gts = _from_yolo(source, names)
    elif source.is_dir():
        gts = _from_pairs(source)
    else:
        raise FileNotFoundError(f"unrecognised ground truth source: {source}")
    if classes is not None:
        keep = set(classes)
        for g in gts:
            sel = [i for i, c in enumerate(g.classes) if c in keep]
            g.masks, g.classes = [g.masks[i] for i in sel], [g.classes[i] for i in sel]
    return gts


def save_predictions(path: str | Path, records: Iterable[dict]) -> Path:
    """Each record: ``{"image": str, "masks": [bool arrays], "scores": [...], "labels": [...]}``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps({
                "image": str(rec["image"]),
                "scores": [round(float(s), 5) for s in rec["scores"]],
                "labels": list(rec.get("labels", [""] * len(rec["scores"]))),
                "rles": [rle_encode(m) for m in rec["masks"]],
                **{k: v for k, v in rec.items() if k not in ("image", "scores", "labels", "masks")},
            }) + "\n")
    return path


def load_predictions(path: str | Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            rec["masks"] = [rle_decode(r) for r in rec.pop("rles")]
            yield rec
