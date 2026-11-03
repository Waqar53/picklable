"""Download the public datasets the evaluation uses, with resume and checksum checks.

* ``ws2``: WS2 test set (Zenodo 14793518, CC BY 4.0, 0.48 GB): 1,497 frames from a
  plastic-sorting plant conveyor, binary masks of the items a human sorter removed. Note
  that its before/after images come from two cameras 2 m apart, so they are not
  pixel-aligned pairs. Here it is a real-conveyor segmentation test set (masks converted to
  instances by connected components), not auto-labeler input.
* ``ws2-train``: WS2 unannotated train/val frames (2.2 GB).
* ``zerowaste-f``: ZeroWaste-f (Zenodo 6412647, 7.5 GB): instance labels for cardboard,
  soft plastic, rigid plastic and metal from a US recycling facility. The project page says
  CC BY-NC 4.0; Zenodo lists CC BY 4.0. Treat it as non-commercial.
* ``biocycling``: no public direct link; prints where to request it.

    picklabel download ws2 --out data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

from picklabel.io import IMAGE_EXTS, write_json

ZENODO = {
    "ws2": ("14793518", ["test_set.zip", "README.md"]),
    "ws2-train": ("14793518", ["train_val_no_backgrounds.zip"]),
    "zerowaste-f": ("6412647", ["zerowaste-f-final.zip"]),
}
BIOCYCLING = ("Project Biocycling (Swiss compost / digestate with plastic foil, hard plastic, glass, aluminium; "
              "RGB + NIR hyperspectral): see https://biocycling-readme.static.hf.space/README.md and the paper "
              "'Detection of Foreign Objects in Compost Fertilizer' (IEEE, doc 11081483) for access.")


def zenodo_files(record: str) -> dict[str, dict]:
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}", timeout=60) as r:
        data = json.load(r)
    return {f["key"]: {"url": f["links"]["self"], "size": f["size"], "checksum": f.get("checksum", "")}
            for f in data.get("files", [])}


def fetch(url: str, dest: Path, size: int | None = None, md5: str | None = None) -> Path:
    """Download with resume (HTTP Range) and an md5 check."""
    from tqdm import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if size is None or have < size:
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "ab" if have else "wb") as f, tqdm(
                total=size, initial=have, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    if md5:
        h = hashlib.md5()
        with open(dest, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        if h.hexdigest() != md5:
            raise SystemExit(f"checksum mismatch for {dest}; delete it and retry")
    return dest


def ws2_to_coco(root: Path, out_json: Path, min_area: int = 64) -> Path:
    """WS2 test masks (binary, per frame) -> COCO instances via connected components."""
    from picklabel.roboflow_sync.coco import CocoBuilder

    coco = CocoBuilder(["plastic"])
    mask_root = next(root.rglob("masks"), None)
    img_root = next(root.rglob("images"), None)
    if mask_root is None or img_root is None:
        raise SystemExit(f"no images/ and masks/ folders under {root}")
    n = 0
    for mpath in sorted((mask_root / "before").rglob("*")):
        if mpath.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = mpath.relative_to(mask_root).with_suffix("")
        img = next((img_root / rel.with_suffix(e) for e in IMAGE_EXTS if (img_root / rel.with_suffix(e)).exists()),
                   None)
        if img is None:
            continue
        m = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE) > 127
        image_id = coco.add_image(str(img.relative_to(out_json.parent)), m.shape[0], m.shape[1])
        k, labels = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
        for i in range(1, k):
            inst = labels == i
            if inst.sum() >= min_area:
                coco.add_instance(image_id, inst, "plastic")
        n += 1
    coco.save(out_json)
    print(f"wrote {n} WS2 test frames to {out_json}")
    return out_json


def find_coco_files(root: Path) -> list[tuple[Path, list[str]]]:
    found = []
    for p in root.rglob("*.json"):
        try:
            with open(p) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and "annotations" in data and "categories" in data:
            found.append((p, [c["name"] for c in data["categories"]]))
    return found


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Download public evaluation datasets.")
    ap.add_argument("name", choices=[*ZENODO, "biocycling"])
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args(argv)
    if args.name == "biocycling":
        print(BIOCYCLING)
        return
    record, wanted = ZENODO[args.name]
    files = zenodo_files(record)
    out = Path(args.out) / args.name.split("-")[0]
    for key in wanted:
        meta = files[key]
        md5 = meta["checksum"].split(":", 1)[1] if meta["checksum"].startswith("md5:") else None
        path = fetch(meta["url"], out / key, meta["size"], md5)
        if path.suffix == ".zip" and not args.no_extract:
            with zipfile.ZipFile(path) as z:
                z.extractall(out)
    if args.name == "ws2" and not args.no_extract:
        ws2_to_coco(out, out / "ws2_test_coco.json")
    for path, cats in find_coco_files(out):
        print(f"COCO file: {path}  classes: {cats}")
    write_json(out / "SOURCE.json", {"dataset": args.name, "zenodo_record": record, "files": wanted})


if __name__ == "__main__":
    main()
