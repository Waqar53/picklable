"""COCO JSON for auto-labels: one file per image (what Roboflow's upload takes) and one per dataset.

PickLabel's provenance (route, quality score, source model, propagated or not) rides along
in each annotation under a ``picklabel`` key, which COCO readers ignore.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from picklabel.io import read_json, write_json
from picklabel.masks import mask_bbox, mask_to_polygons, polygons_to_mask


class CocoBuilder:
    """Accumulates images and instance masks into a COCO dict."""

    def __init__(self, categories: Iterable[str] = ("plastic",)):
        self.categories: list[str] = []
        for c in categories:
            self.category_id(c)
        self.images: list[dict] = []
        self.annotations: list[dict] = []

    def category_id(self, name: str) -> int:
        if name not in self.categories:
            self.categories.append(name)
        return self.categories.index(name) + 1

    def add_image(self, file_name: str, height: int, width: int, extra: dict | None = None) -> int:
        image_id = len(self.images) + 1
        rec = {"id": image_id, "file_name": file_name, "height": int(height), "width": int(width)}
        if extra:
            rec["picklabel"] = extra
        self.images.append(rec)
        return image_id

    def add_instance(self, image_id: int, mask: np.ndarray, class_name: str, extra: dict | None = None,
                     epsilon: float = 1.0) -> int | None:
        polys = mask_to_polygons(mask, min_area=16, epsilon=epsilon)
        box = mask_bbox(mask)
        if not polys or box is None:
            return None
        ann_id = len(self.annotations) + 1
        x1, y1, x2, y2 = box
        ann = {
            "id": ann_id, "image_id": image_id, "category_id": self.category_id(class_name),
            "segmentation": polys, "bbox": [x1, y1, x2 - x1, y2 - y1], "area": int(mask.sum()), "iscrowd": 0,
        }
        if extra:
            ann["picklabel"] = extra
        self.annotations.append(ann)
        return ann_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "info": {"description": "PickLabel auto-labels"},
            "images": self.images,
            "annotations": self.annotations,
            "categories": [{"id": i + 1, "name": n, "supercategory": "waste"} for i, n in enumerate(self.categories)],
        }

    def save(self, path: str | Path) -> Path:
        return write_json(path, self.to_dict())


def write_image_coco(path: str | Path, file_name: str, height: int, width: int,
                     instances: Iterable[tuple[np.ndarray, str, dict]], categories: Iterable[str]) -> Path:
    """Single-image COCO file whose ``file_name`` matches the uploaded image."""
    b = CocoBuilder(categories)
    image_id = b.add_image(file_name, height, width)
    for mask, cls, extra in instances:
        b.add_instance(image_id, mask, cls, extra)
    return b.save(path)


def load_coco(path: str | Path, class_filter: Iterable[str] | None = None
              ) -> dict[str, tuple[int, int, list[tuple[np.ndarray, str, dict]]]]:
    """Decode a COCO file into ``{file_name: (h, w, [(mask, class, ann), ...])}``.

    Polygon and uncompressed-RLE segmentations are supported.
    """
    from picklabel.masks import rle_decode

    data = read_json(path)
    names = {c["id"]: c["name"] for c in data.get("categories", [])}
    keep = set(class_filter) if class_filter is not None else None
    out: dict[str, tuple[int, int, list]] = {}
    by_id = {}
    for img in data.get("images", []):
        out[img["file_name"]] = (img["height"], img["width"], [])
        by_id[img["id"]] = img
    for ann in data.get("annotations", []):
        img = by_id.get(ann["image_id"])
        cls = names.get(ann["category_id"], str(ann["category_id"]))
        if img is None or (keep is not None and cls not in keep):
            continue
        h, w = img["height"], img["width"]
        seg = ann.get("segmentation")
        if isinstance(seg, list) and seg:
            mask = polygons_to_mask(seg, h, w)
        elif isinstance(seg, dict) and isinstance(seg.get("counts"), list):
            mask = rle_decode(seg)
        else:
            continue  # boxes only or compressed RLE: not usable as a mask here
        out[img["file_name"]][2].append((mask, cls, ann))
    return out
