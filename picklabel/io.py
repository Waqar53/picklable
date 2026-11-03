"""Reading and writing pick pairs on disk.

Canonical layout, one folder per pick::

    <root>/<session>/<pick_id>/before.jpg
                              after.jpg
                              pick.json         # grasp_px, class_name, grasp_success, belt_shift_px, ...
                              gt_mask.png       # optional: gold mask of the picked item
                              gt_instances.png  # optional: 16-bit id map of every target in `before`
    <root>/<session>/session.json              # optional: {"cleared": true, "final_after": "final_after.jpg"}

The flat layout of the first rig script (``<id>_before.jpg``, ``<id>_after.jpg`` and
``<id>.json`` holding ``pick_xy``) is read too.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from picklabel.types import PickPair

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_KNOWN_KEYS = {
    "pick_id", "session", "grasp_px", "pick_xy", "class_name", "label",
    "grasp_success", "belt_shift_px", "timestamp",
}


def read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def write_image(path: str | Path, image: np.ndarray, quality: int = 95) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if path.suffix.lower() in (".jpg", ".jpeg") else []
    if not cv2.imwrite(str(path), image, params):
        raise OSError(f"Could not write image: {path}")
    return path


def read_mask(path: str | Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return m > 127


def read_id_map(path: str | Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Could not read id map: {path}")
    return m.astype(np.uint16)


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def _find_image(folder: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _pair_from_meta(meta: dict, before: Path, after: Path, folder: Path, default_id: str,
                    default_session: str) -> PickPair:
    grasp = meta.get("grasp_px", meta.get("pick_xy"))
    if grasp is None:
        raise ValueError(f"{folder}: pick metadata has no grasp_px / pick_xy")
    shift = meta.get("belt_shift_px")
    pair = PickPair(
        pick_id=str(meta.get("pick_id", default_id)),
        before=read_image(before),
        after=read_image(after),
        grasp_xy=(float(grasp[0]), float(grasp[1])),
        session=str(meta.get("session", default_session)),
        class_name=str(meta.get("class_name", meta.get("label", "plastic"))),
        grasp_success=meta.get("grasp_success"),
        belt_shift_px=(float(shift[0]), float(shift[1])) if shift is not None else None,
        timestamp=meta.get("timestamp"),
        meta={k: v for k, v in meta.items() if k not in _KNOWN_KEYS},
        path=folder,
    )
    return pair


def load_pair(pick_dir: str | Path) -> PickPair:
    """Load one pick folder in the canonical layout."""
    folder = Path(pick_dir)
    before, after = _find_image(folder, "before"), _find_image(folder, "after")
    if before is None or after is None:
        raise FileNotFoundError(f"{folder}: needs before.* and after.* images")
    meta = read_json(folder / "pick.json") if (folder / "pick.json").exists() else {}
    pair = _pair_from_meta(meta, before, after, folder, folder.name, folder.parent.name)
    if (folder / "gt_mask.png").exists():
        pair.gt_mask = read_mask(folder / "gt_mask.png")
    if (folder / "gt_instances.png").exists():
        pair.gt_instances = read_id_map(folder / "gt_instances.png")
    return pair


def save_pair(pair: PickPair, root: str | Path, image_ext: str = ".jpg", quality: int = 95) -> Path:
    """Write a pair in the canonical layout and return its folder."""
    folder = Path(root) / pair.session / pair.pick_id
    folder.mkdir(parents=True, exist_ok=True)
    write_image(folder / f"before{image_ext}", pair.before, quality)
    write_image(folder / f"after{image_ext}", pair.after, quality)
    meta = {
        "pick_id": pair.pick_id,
        "session": pair.session,
        "grasp_px": [round(float(v), 2) for v in pair.grasp_xy],
        "class_name": pair.class_name,
        "grasp_success": pair.grasp_success,
        "belt_shift_px": list(pair.belt_shift_px) if pair.belt_shift_px is not None else None,
        "timestamp": pair.timestamp,
        **pair.meta,
    }
    write_json(folder / "pick.json", meta)
    if pair.gt_mask is not None:
        cv2.imwrite(str(folder / "gt_mask.png"), pair.gt_mask.astype(np.uint8) * 255)
    if pair.gt_instances is not None:
        cv2.imwrite(str(folder / "gt_instances.png"), pair.gt_instances.astype(np.uint16))
    return folder


def _sort_key(folder: Path) -> tuple:
    ts = None
    meta_path = folder / "pick.json"
    if meta_path.exists():
        try:
            ts = read_json(meta_path).get("timestamp")
        except (OSError, json.JSONDecodeError):
            ts = None
    return (ts if isinstance(ts, (int, float)) else float("inf"), folder.name)


def find_pick_dirs(root: str | Path) -> list[Path]:
    """All canonical pick folders under ``root``, grouped by session and in pick order."""
    root = Path(root)
    dirs = {p.parent for p in root.rglob("pick.json")}
    dirs |= {p.parent for p in root.rglob("before.*") if p.suffix.lower() in IMAGE_EXTS}
    by_session: dict[Path, list[Path]] = {}
    for d in dirs:
        if _find_image(d, "before") and _find_image(d, "after"):
            by_session.setdefault(d.parent, []).append(d)
    out: list[Path] = []
    for session in sorted(by_session):
        out.extend(sorted(by_session[session], key=_sort_key))
    return out


def _iter_flat(root: Path) -> Iterator[PickPair]:
    for meta_path in sorted(root.glob("*.json")):
        stem = meta_path.stem
        before, after = _find_image(root, f"{stem}_before"), _find_image(root, f"{stem}_after")
        if before is None or after is None:
            continue
        meta = read_json(meta_path)
        yield _pair_from_meta(meta, before, after, root, stem, root.name)


def load_pairs(root: str | Path, limit: int | None = None) -> Iterator[PickPair]:
    """Yield every pair under ``root`` (canonical layout first, then flat files)."""
    root = Path(root)
    n = 0
    for folder in find_pick_dirs(root):
        if limit is not None and n >= limit:
            return
        yield load_pair(folder)
        n += 1
    for pair in _iter_flat(root):
        if limit is not None and n >= limit:
            return
        yield pair
        n += 1


def load_session_meta(session_dir: str | Path) -> dict:
    path = Path(session_dir) / "session.json"
    return read_json(path) if path.exists() else {}
