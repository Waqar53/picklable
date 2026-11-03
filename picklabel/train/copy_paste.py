"""Copy-paste augmentation: labelled plastic pasted onto clean organic backgrounds.

Instances come from a YOLO-seg dataset's train split. Backgrounds are ideally the cleared-tray
frames PickLabel already produces (organics and look-alikes, no plastic). Each output image
gets a few pastes with random scale, rotation, flip and brightness and feathered edges. Later
pastes occlude earlier ones and the labels are clipped to what stays visible. Results go into
the train split with a ``cp_`` prefix, so val and test stay untouched.

The ZeroWaste authors report that object-level augmentation helps rare classes; here it
targets the rare, hard cases (clear film, small fragments).

    picklabel copy-paste --dataset datasets/R3 --backgrounds "runs/autolabel/*/images/*final_after*" --n 300
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

from picklabel.io import IMAGE_EXTS, read_image, write_image
from picklabel.masks import largest_component, mask_to_polygons, polygons_to_mask


def load_instances(dataset: Path, split: str = "train", max_instances: int = 800, seed: int = 0
                   ) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """(BGR crop, bool mask crop, class id) for each labelled instance."""
    rng = np.random.default_rng(seed)
    out = []
    images = sorted(p for p in (dataset / "images" / split).iterdir()
                    if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("cp_"))
    for img_path in images:
        txt = dataset / "labels" / split / f"{img_path.stem}.txt"
        if not txt.exists() or not txt.read_text().strip():
            continue
        img = read_image(img_path)
        h, w = img.shape[:2]
        for line in txt.read_text().splitlines():
            vals = line.split()
            if len(vals) < 7:
                continue
            xy = (np.array(vals[1:], np.float64).reshape(-1, 2) * [w, h]).reshape(-1).tolist()
            m = polygons_to_mask([xy], h, w)
            ys, xs = np.nonzero(m)
            if ys.size < 64:
                continue
            y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            out.append((img[y1:y2, x1:x2].copy(), m[y1:y2, x1:x2].copy(), int(vals[0])))
    rng.shuffle(out)
    return out[:max_instances]


def _transform(crop: np.ndarray, mask: np.ndarray, rng: np.random.Generator, scale_range=(0.6, 1.3)):
    s = rng.uniform(*scale_range)
    angle = rng.uniform(0, 360)
    h, w = mask.shape
    side = int(np.ceil(np.hypot(h, w) * s)) + 2
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, s)
    M[:, 2] += (side / 2 - w / 2, side / 2 - h / 2)
    c = cv2.warpAffine(crop, M, (side, side), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    m = cv2.warpAffine(mask.astype(np.uint8), M, (side, side), flags=cv2.INTER_NEAREST) > 0
    if rng.random() < 0.5:
        c, m = c[:, ::-1], m[:, ::-1]
    c = np.clip(c.astype(np.float32) * rng.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
    return c, m


def paste(bg: np.ndarray, instances, rng: np.random.Generator, n_range=(1, 4)
          ) -> tuple[np.ndarray, list[tuple[np.ndarray, int]]]:
    out = bg.copy()
    h, w = bg.shape[:2]
    placed: list[tuple[np.ndarray, int, int]] = []  # (mask, class, original area)
    for _ in range(int(rng.integers(n_range[0], n_range[1] + 1))):
        crop, mask, cls = instances[int(rng.integers(len(instances)))]
        c, m = _transform(crop, mask, rng)
        ch, cw = m.shape
        if ch >= h or cw >= w or m.sum() < 64:
            continue
        y, x = int(rng.integers(0, h - ch)), int(rng.integers(0, w - cw))
        full = np.zeros((h, w), bool)
        full[y:y + ch, x:x + cw] = m
        alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)[..., None]
        region = out[y:y + ch, x:x + cw].astype(np.float32)
        out[y:y + ch, x:x + cw] = (region * (1 - alpha) + c.astype(np.float32) * alpha).astype(np.uint8)
        placed = [(pm & ~full, pc, pa) for pm, pc, pa in placed]
        placed.append((full, cls, int(full.sum())))
    labels = [(largest_component(m), c) for m, c, a in placed if m.sum() >= 0.3 * a]
    return out, labels


def augment(dataset: Path, backgrounds: list[Path], n: int, seed: int = 0) -> int:
    rng = np.random.default_rng(seed)
    instances = load_instances(dataset, seed=seed)
    if not instances or not backgrounds:
        raise SystemExit("need labelled instances in the train split and at least one background image")
    written = 0
    for i in range(n):
        bg = read_image(backgrounds[int(rng.integers(len(backgrounds)))])
        img, labels = paste(bg, instances, rng)
        if not labels:
            continue
        h, w = img.shape[:2]
        lines = []
        for m, cls in labels:
            polys = mask_to_polygons(m)
            if polys:
                xy = (np.asarray(polys[0]).reshape(-1, 2) / [w, h]).reshape(-1)
                lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in np.clip(xy, 0, 1)))
        write_image(dataset / "images" / "train" / f"cp_{i:05d}.jpg", img)
        (dataset / "labels" / "train" / f"cp_{i:05d}.txt").write_text("\n".join(lines) + "\n")
        written += 1
    return written


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Copy-paste augmentation into a YOLO-seg dataset's train split.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backgrounds", required=True, help="glob of background images (cleared trays)")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    bgs = [Path(p) for p in sorted(glob.glob(args.backgrounds))]
    n = augment(Path(args.dataset), bgs, args.n, args.seed)
    print(f"wrote {n} copy-paste images into {args.dataset}/images/train")


if __name__ == "__main__":
    main()
