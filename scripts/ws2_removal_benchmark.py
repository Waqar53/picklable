"""Real conveyor images, simulated removal: does PickLabel find the item that was taken?

WS2 (Marelli et al., ICCV 2025 Workshops, CC BY 4.0) has real frames from a plastic-sorting
plant, with hand-made masks of the items a human sorter removes. Its before and after frames
come from two different cameras, so they are not pairs. This script makes pairs from the
real "before" frames:

* one annotated item (a connected component of the mask) is the pick;
* AFTER = the same frame with that item removed (OpenCV inpainting + matched sensor grain),
  re-captured with camera shake, exposure / white-balance drift, noise and JPEG. In half the
  picks, annotated items touching it are nudged by a few pixels, as pulling an item does;
* GRASP = an interior point of the item plus a few pixels of noise;
* ~10% are missed picks: AFTER is a re-capture with nothing removed.

Ground truth is the real hand-made mask of the removed item. Textures, lighting, translucent
PET and clutter are real; only the removal is simulated, and results are labelled that way.

    python scripts/ws2_removal_benchmark.py ~/picklabel-data/ws2 --out ~/picklabel-data/ws2_pairs --n 120
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from picklabel.io import IMAGE_EXTS, save_pair  # noqa: E402
from picklabel.types import PickPair  # noqa: E402


def find_frames(root: Path) -> list[tuple[Path, Path]]:
    mask_root = next(p for p in root.rglob("masks") if (p / "before").exists())
    img_root = mask_root.parent / "images"
    pairs = []
    for m in sorted((mask_root / "before").rglob("*")):
        if m.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = m.relative_to(mask_root).with_suffix("")
        img = next((img_root / rel.with_suffix(e) for e in IMAGE_EXTS if (img_root / rel.with_suffix(e)).exists()), None)
        if img is not None:
            pairs.append((img, m))
    return pairs


def components(mask: np.ndarray) -> list[np.ndarray]:
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return [labels == i for i in range(1, n)]


def grain_fill(img: np.ndarray, region: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Inpaint ``region`` and add sensor grain matched to the surroundings, so the fill isn't glassy."""
    out = cv2.inpaint(img, region.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
    ring = cv2.dilate(region.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~region
    hf = img.astype(np.float32) - cv2.GaussianBlur(img, (0, 0), 2).astype(np.float32)
    sigma = float(hf[ring].std()) if ring.any() else 4.0
    grain = cv2.GaussianBlur(rng.normal(0, 1, img.shape).astype(np.float32), (0, 0), 1.0)
    grain *= sigma / (grain.std() + 1e-6)
    filled = out.astype(np.float32)
    filled[region] += grain[region]
    return np.clip(filled, 0, 255).astype(np.uint8)


def nudge(img: np.ndarray, item: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shift one item by a few pixels, filling the strip it leaves."""
    d = rng.integers(2, 5, 2) * rng.choice([-1, 1], 2)
    M = np.float32([[1, 0, d[0]], [0, 1, d[1]]])
    h, w = item.shape
    moved = cv2.warpAffine(item.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
    shifted = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = grain_fill(img, item & ~moved, rng)
    out[moved] = shifted[moved]
    return out


def capture(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Re-capture: shake, exposure and white-balance drift, sensor noise, JPEG."""
    h, w = img.shape[:2]
    a = np.deg2rad(rng.normal(0, 0.25))
    tx, ty = rng.normal(0, 1.5, 2)
    c, s = np.cos(a), np.sin(a)
    M = np.float32([[c, -s, (1 - c) * w / 2 + s * h / 2 + tx], [s, c, (1 - c) * h / 2 - s * w / 2 + ty]])
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    out *= (1 + rng.normal(0, 0.04)) * (1 + rng.normal(0, 0.03, 3))
    out += rng.normal(0, 2.0, out.shape)
    ok, enc = cv2.imencode(".jpg", np.clip(out, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def grasp_point(item: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    dist = cv2.distanceTransform(item.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(dist >= 0.5 * dist.max())
    i = int(rng.integers(len(ys)))
    return float(xs[i] + rng.normal(0, 2)), float(ys[i] + rng.normal(0, 2))


def make_pairs(root: Path, out: Path, n: int, seed: int, p_missed: float = 0.1, p_nudge: float = 0.5,
               min_frac: float = 0.003, max_frac: float = 0.12) -> list[dict]:
    rng = np.random.default_rng(seed)
    frames = find_frames(root)
    order = rng.permutation(len(frames))
    records = []
    for idx in order:
        if len(records) >= n:
            break
        img_path, mask_path = frames[idx]
        before = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        mask = raw > (127 if raw.max() > 1 else 0)  # masks may be stored 0/255 or 0/1
        if before is None or mask.shape != before.shape[:2]:
            continue
        h, w = mask.shape
        comps = [c for c in components(mask) if min_frac * h * w <= c.sum() <= max_frac * h * w]
        # keep items fully inside the frame so the whole removal is observable
        comps = [c for c in comps if not (c[:12].any() or c[-12:].any() or c[:, :12].any() or c[:, -12:].any())]
        if not comps:
            continue
        item = comps[int(rng.integers(len(comps)))]
        missed = bool(rng.random() < p_missed)
        scene = before.copy()
        nudged = 0
        if not missed:
            near = cv2.dilate(item.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
            if rng.random() < p_nudge:
                for other in components(mask & ~item):
                    if (other & near).any():
                        scene = nudge(scene, other, rng)
                        nudged += 1
            scene = grain_fill(scene, cv2.dilate(item.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool), rng)
        after = capture(scene, rng)
        grasp = grasp_point(item, rng)
        pick_id = f"{len(records):03d}"
        session = f"ws2-{img_path.parent.name}-{img_path.stem}"
        meta = {"synthetic": False, "source": "WS2 test set (real), simulated removal", "frame": str(img_path.name),
                "missed": missed, "nudged_neighbours": nudged, "item_area_frac": round(float(item.sum()) / (h * w), 5)}
        save_pair(PickPair(pick_id=pick_id, before=before, after=after, grasp_xy=grasp, session=session,
                           grasp_success=(False if missed and rng.random() < 0.5 else None),
                           gt_mask=item, meta=meta), out, quality=95)
        records.append({"session": session, "pick_id": pick_id, **meta})
    (out / "benchmark.json").write_text(json.dumps({"source": str(root), "seed": seed, "picks": records}, indent=1))
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Make before/after pick pairs from real WS2 conveyor frames.")
    ap.add_argument("ws2_root", help="folder holding the extracted WS2 test set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    recs = make_pairs(Path(args.ws2_root), Path(args.out), args.n, args.seed)
    print(f"{len(recs)} picks ({sum(r['missed'] for r in recs)} missed, "
          f"{sum(r['nudged_neighbours'] > 0 for r in recs)} with nudged neighbours) -> {args.out}")


if __name__ == "__main__":
    main()
