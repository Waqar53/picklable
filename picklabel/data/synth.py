"""Synthetic pick sessions with exact ground truth.

A session is a tray of wet organic waste with a few plastic contaminants. A simulated robot
removes the plastics one at a time, most visible first. Each pick yields a before/after
pair carrying the nuisances of a real cell: camera shake, exposure and white-balance drift,
sensor noise, JPEG, neighbours that shift when an item is pulled, see-through film, dark
plastic on dark waste and the occasional missed grasp. In ``belt`` mode the content also
translates between the two frames and the "encoder" reports the shift with a little error.

Ground truth per pick: the visible mask of the picked item in the before frame, and an id
map of every plastic still present. This exists to test the pipeline end to end and to make
regressions visible. Numbers measured on it say nothing about real waste.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from picklabel.io import save_pair, write_image, write_json
from picklabel.types import PickPair

ORGANIC_BGR = [
    (40, 70, 110), (30, 55, 85), (45, 95, 60), (60, 170, 210), (40, 125, 215),
    (55, 140, 75), (200, 208, 214), (28, 42, 62), (70, 110, 150), (35, 80, 45),
]
PLASTIC_BGR = [
    (185, 95, 25), (228, 228, 230), (40, 40, 200), (60, 150, 40), (40, 200, 230), (200, 160, 120),
]


@dataclass
class SynthConfig:
    width: int = 960
    height: int = 720
    n_organic: tuple[int, int] = (35, 60)
    n_plastic: tuple[int, int] = (3, 6)
    p_film: float = 0.3
    p_dark: float = 0.15
    p_missed: float = 0.07
    p_neighbor_shift: float = 0.4
    shake_px: float = 3.0
    shake_deg: float = 0.5
    exposure_drift: float = 0.05
    wb_drift: float = 0.03
    noise_sigma: float = 2.0
    jpeg_quality: int = 90
    report_gripper: float = 0.5
    belt: bool = False
    belt_shift: tuple[float, float] = (40.0, 140.0)
    encoder_noise_px: float = 2.0


@dataclass
class Item:
    kind: str  # organic | plastic
    x: int
    y: int
    mask: np.ndarray  # local bool mask
    tex: np.ndarray  # local float32 BGR
    alpha: float = 1.0
    film: bool = False
    dark: bool = False
    uid: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def opaque(self) -> bool:
        return self.alpha >= 0.9


def _blob(rng: np.random.Generator, r: float, aspect: float, roughness: float) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    radius = np.ones_like(theta)
    for k in range(2, 8):
        radius += rng.normal(0, roughness / k) * np.cos(k * theta + rng.uniform(0, 2 * np.pi))
    radius = np.clip(radius, 0.35, 1.8) * r
    x, y = radius * np.cos(theta) * aspect, radius * np.sin(theta)
    a = rng.uniform(0, np.pi)
    x, y = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
    half = int(np.ceil(max(np.abs(x).max(), np.abs(y).max()))) + 3
    pts = np.stack([x + half, y + half], 1).round().astype(np.int32)
    m = np.zeros((2 * half + 1, 2 * half + 1), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def _noise(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    n = rng.normal(0, 1, shape).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma) if sigma > 0 else n
    return n / (n.std() + 1e-6)


def _organic(rng: np.random.Generator, cfg: SynthConfig) -> Item:
    r = rng.uniform(10, 45)
    m = _blob(rng, r, rng.uniform(1.0, 2.8), rng.uniform(0.15, 0.45))
    base = np.array(ORGANIC_BGR[rng.integers(len(ORGANIC_BGR))], np.float32) * rng.uniform(0.75, 1.15)
    shape = m.shape
    tex = base + 22 * _noise(rng, shape, 5)[..., None] + 9 * _noise(rng, shape, 0.8)[..., None]
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    tex *= (0.65 + 0.35 * np.clip(dist / 6.0, 0, 1))[..., None]  # darker, wet-looking rims
    return Item("organic", 0, 0, m, tex.astype(np.float32))


def _plastic(rng: np.random.Generator, cfg: SynthConfig) -> Item:
    film = rng.random() < cfg.p_film
    dark = (not film) and rng.random() < cfg.p_dark
    r = rng.uniform(28, 70)
    m = _blob(rng, r, rng.uniform(1.0, 2.2), rng.uniform(0.3, 0.6))
    shape = m.shape
    if film:
        base, alpha = np.array([232, 236, 240], np.float32), float(rng.uniform(0.35, 0.55))
    elif dark:
        base, alpha = np.array([28, 28, 31], np.float32), 1.0
    else:
        base, alpha = np.array(PLASTIC_BGR[rng.integers(len(PLASTIC_BGR))], np.float32), 1.0
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    ang, freq = rng.uniform(0, np.pi), rng.uniform(0.08, 0.2)
    wrinkles = np.sin((xx * np.cos(ang) + yy * np.sin(ang)) * freq + 3 * _noise(rng, shape, 6))
    tex = base * (1 + 0.08 * wrinkles)[..., None] + 6 * _noise(rng, shape, 1.5)[..., None]
    highlights = np.zeros(shape, np.float32)
    for _ in range(rng.integers(2, 6)):
        p0 = rng.uniform(0, shape[1]), rng.uniform(0, shape[0])
        p1 = p0[0] + rng.normal(0, r), p0[1] + rng.normal(0, r)
        cv2.line(highlights, tuple(int(v) for v in p0), tuple(int(v) for v in p1), 1.0, int(rng.integers(1, 4)))
    highlights = cv2.GaussianBlur(highlights, (0, 0), 1.5)
    tex = tex + (255 - tex) * np.clip(highlights * (0.9 if film else 0.6), 0, 1)[..., None]
    return Item("plastic", 0, 0, m, tex.astype(np.float32), alpha=alpha, film=film, dark=dark)


def _tray(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    base = np.array([38, 36, 40], np.float32)
    canvas = base + 8 * _noise(rng, (h, w), 40)[..., None] + 3 * _noise(rng, (h, w), 1)[..., None]
    dust = rng.random((h, w)) < 0.004
    canvas[dust] = (22, 30, 42)
    return canvas


def _place(item: Item, rng: np.random.Generator, w: int, h: int) -> Item:
    mh, mw = item.mask.shape
    item.x = int(rng.integers(-mw // 4, max(1, w - 3 * mw // 4)))
    item.y = int(rng.integers(-mh // 4, max(1, h - 3 * mh // 4)))
    return item


def _slices(item: Item, w: int, h: int):
    mh, mw = item.mask.shape
    x1, y1, x2, y2 = max(item.x, 0), max(item.y, 0), min(item.x + mw, w), min(item.y + mh, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return (slice(y1, y2), slice(x1, x2)), (slice(y1 - item.y, y2 - item.y), slice(x1 - item.x, x2 - item.x))


def render(items: list[Item], tray: np.ndarray) -> np.ndarray:
    canvas = tray.copy()
    h, w = canvas.shape[:2]
    for it in items:
        sl = _slices(it, w, h)
        if sl is None:
            continue
        (gy, gx), (ly, lx) = sl
        m = it.mask[ly, lx].astype(np.float32)
        # Soft contact shadow, offset down-right, then the item itself.
        shadow = cv2.GaussianBlur(m, (0, 0), 2.0)
        canvas[gy, gx] *= (1 - 0.25 * shadow * it.alpha)[..., None]
        a = (m * it.alpha)[..., None]
        canvas[gy, gx] = canvas[gy, gx] * (1 - a) + it.tex[ly, lx] * a
    return canvas


def full_mask(item: Item, w: int, h: int) -> np.ndarray:
    out = np.zeros((h, w), bool)
    sl = _slices(item, w, h)
    if sl is not None:
        (gy, gx), (ly, lx) = sl
        out[gy, gx] = item.mask[ly, lx]
    return out


def visible_mask(items: list[Item], idx: int, w: int, h: int) -> np.ndarray:
    """Pixels of item ``idx`` not covered by an opaque item above it."""
    vis = full_mask(items[idx], w, h)
    for above in items[idx + 1:]:
        if above.opaque:
            vis &= ~full_mask(above, w, h)
    return vis


def _capture(canvas: np.ndarray, rng: np.random.Generator, cfg: SynthConfig, x0: int = 0
             ) -> tuple[np.ndarray, np.ndarray]:
    """Crop the camera window at ``x0``, add shake, lighting drift, noise and JPEG."""
    w, h = cfg.width, cfg.height
    ang = np.deg2rad(rng.normal(0, cfg.shake_deg / 2))
    tx, ty = rng.normal(0, cfg.shake_px / 2, 2)
    c, s = np.cos(ang), np.sin(ang)
    cx, cy = w / 2, h / 2
    # Maps canvas coordinates (shifted by the window origin) to image coordinates.
    A = np.array([[c, -s, cx - c * cx + s * cy + tx - x0 * c], [s, c, cy - s * cx - c * cy + ty - x0 * s]])
    img = cv2.warpAffine(canvas, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    gain = 1 + rng.normal(0, cfg.exposure_drift)
    wb = 1 + rng.normal(0, cfg.wb_drift, 3)
    img = img * gain * wb + rng.normal(0, cfg.noise_sigma, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), A


def _warp_mask(mask: np.ndarray, A: np.ndarray, cfg: SynthConfig) -> np.ndarray:
    return cv2.warpAffine(mask.astype(np.uint8), A, (cfg.width, cfg.height), flags=cv2.INTER_NEAREST) > 0


def _grasp_point(mask: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(dist >= 0.5 * dist.max())
    i = int(rng.integers(len(ys)))
    return float(xs[i] + rng.normal(0, 1.5)), float(ys[i] + rng.normal(0, 1.5))


def make_session(session_id: str, rng: np.random.Generator, cfg: SynthConfig
                 ) -> tuple[list[PickPair], np.ndarray, bool]:
    """Generate one session: its pairs, the final after image, and whether all plastic is gone.

    In belt mode every pick sees a fresh window of the belt, so frames are not a sequence.
    """
    margin = int(cfg.belt_shift[1]) + 40 if cfg.belt else 0
    tw, th = cfg.width + 2 * margin, cfg.height
    tray = _tray(rng, tw, th)
    organics = [_place(_organic(rng, cfg), rng, tw, th) for _ in range(rng.integers(*cfg.n_organic))]
    plastics = [_place(_plastic(rng, cfg), rng, tw, th) for _ in range(rng.integers(*cfg.n_plastic))]
    if cfg.belt:
        for p in plastics:  # keep contaminants inside the first camera window
            p.x = int(np.clip(p.x, margin, margin + cfg.width - p.mask.shape[1]))
    n_top = len(organics) // 5
    items = organics[:-n_top] + plastics + organics[-n_top:]  # some plastic is partly buried
    for i, it in enumerate(items):
        it.uid = i + 1

    pairs: list[PickPair] = []
    x0 = margin
    k = 0
    last_after = None
    while any(it.kind == "plastic" for it in items) and k < 4 * cfg.n_plastic[1]:
        idx_plastic = [i for i, it in enumerate(items) if it.kind == "plastic"]
        vis = {i: visible_mask(items, i, tw, th) for i in idx_plastic}
        frac = {i: vis[i].sum() / max(full_mask(items[i], tw, th).sum(), 1) for i in idx_plastic}
        target = max(idx_plastic, key=lambda i: (frac[i], i))
        if vis[target].sum() < 50:  # nothing graspable left
            break

        before, A_b = _capture(render(items, tray), rng, cfg, x0)
        gt_mask = _warp_mask(vis[target], A_b, cfg)
        gt_ids = np.zeros((cfg.height, cfg.width), np.uint16)
        for n, i in enumerate(idx_plastic, start=1):
            gt_ids[_warp_mask(vis[i], A_b, cfg) & (gt_ids == 0)] = n
        if gt_mask.sum() < 50:
            break
        grasp = _grasp_point(gt_mask, rng)

        missed = bool(rng.random() < cfg.p_missed)
        target_item = items[target]
        if not missed:
            items.pop(target)
            tm = full_mask(target_item, tw, th)
            near = cv2.dilate(tm.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            for it in items:
                if (full_mask(it, tw, th) & near).any() and rng.random() < cfg.p_neighbor_shift:
                    it.x += int(rng.integers(-6, 7))
                    it.y += int(rng.integers(-6, 7))

        shift = None
        x_after = x0
        if cfg.belt:
            d = float(rng.uniform(*cfg.belt_shift))
            x_after = x0 - int(round(d))  # content moves +d in the image
            shift = (d + float(rng.normal(0, cfg.encoder_noise_px)), float(rng.normal(0, cfg.encoder_noise_px)))
        after, _ = _capture(render(items, tray), rng, cfg, x_after)
        last_after = after

        report = rng.random() < cfg.report_gripper
        pairs.append(PickPair(
            pick_id=f"{k:03d}", before=before, after=after, grasp_xy=grasp, session=session_id,
            class_name="plastic", grasp_success=(not missed) if report else None, belt_shift_px=shift,
            timestamp=float(k), gt_mask=gt_mask, gt_instances=gt_ids,
            meta={"synthetic": True, "missed": missed, "film": target_item.film, "dark": target_item.dark},
        ))
        k += 1
    final_after = last_after if last_after is not None else _capture(render(items, tray), rng, cfg, x0)[0]
    cleared = not cfg.belt and not any(it.kind == "plastic" for it in items)
    return pairs, final_after, cleared


def generate(out_dir: str | Path, n_sessions: int = 8, seed: int = 0, cfg: SynthConfig | None = None,
             prefix: str = "synth") -> list[Path]:
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    sessions = []
    for s in range(n_sessions):
        sid = f"{prefix}{s:03d}"
        pairs, final_after, cleared = make_session(sid, rng, cfg)
        for p in pairs:
            save_pair(p, out_dir, quality=95)
        session_dir = out_dir / sid
        write_image(session_dir / "final_after.jpg", final_after)
        write_json(session_dir / "session.json", {
            "session": sid, "cleared": cleared, "final_after": "final_after.jpg", "synthetic": True,
            "mode": "belt" if cfg.belt else "static", "n_picks": len(pairs),
        })
        sessions.append(session_dir)
    return sessions


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic pick sessions with exact ground truth.")
    ap.add_argument("--out", default="data/synth/static")
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--belt", action="store_true", help="content moves with a conveyor between frames")
    ap.add_argument("--prefix", default=None)
    args = ap.parse_args(argv)
    cfg = SynthConfig(belt=args.belt)
    prefix = args.prefix or ("belt" if args.belt else "tray")
    dirs = generate(args.out, args.sessions, args.seed, cfg, prefix=prefix)
    n = sum(len(list(d.glob("*/pick.json"))) for d in dirs)
    print(f"wrote {n} picks in {len(dirs)} sessions to {args.out}")


if __name__ == "__main__":
    main()
