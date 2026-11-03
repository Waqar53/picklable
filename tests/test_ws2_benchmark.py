"""The real-image benchmark generator, on a tiny WS2-shaped folder."""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from picklabel.autolabel.pipeline import AutoLabeler
from picklabel.io import find_pick_dirs, load_pair
from picklabel.masks import mask_iou

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ws2_removal_benchmark.py"


def load_script():
    spec = importlib.util.spec_from_file_location("ws2_bench", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_benchmark_pairs_are_valid_and_labelable(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "test_set"
    for k in range(4):
        (root / "images" / "before" / "v0").mkdir(parents=True, exist_ok=True)
        (root / "masks" / "before" / "v0").mkdir(parents=True, exist_ok=True)
        img = cv2.GaussianBlur((rng.random((240, 320, 3)) * 255).astype(np.uint8), (0, 0), 3)
        mask = np.zeros((240, 320), np.uint8)
        cv2.ellipse(mask, (120, 120), (40, 25), 20, 0, 360, 1, -1)  # stored as 0/1, like some mask exports
        cv2.circle(mask, (230, 140), 22, 1, -1)
        img[mask > 0] = (40, 170, 220)
        cv2.imwrite(str(root / "images" / "before" / "v0" / f"f{k}.jpg"), img)
        cv2.imwrite(str(root / "masks" / "before" / "v0" / f"f{k}.png"), mask)
    mod = load_script()
    recs = mod.make_pairs(tmp_path, tmp_path / "pairs", n=4, seed=1, p_missed=0.0, min_frac=0.005)
    assert len(recs) == 4
    meta = json.loads((tmp_path / "pairs" / "benchmark.json").read_text())
    assert len(meta["picks"]) == 4
    labeler = AutoLabeler()
    ious = [mask_iou(labeler.label(p).mask, p.gt_mask) for p in map(load_pair, find_pick_dirs(tmp_path / "pairs"))]
    assert min(ious) > 0.6, ious
