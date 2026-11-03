"""Ablation: turn each robustness piece of the auto-labeler off and re-measure gold IoU.

Runs on any pairs folder with gold masks (synthetic, or your hand-labelled gold set):

    python scripts/ablate_autolabel.py data/gold --out report/autolabel/ablation.md
"""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from picklabel.autolabel.pipeline import AutoLabelConfig, AutoLabeler  # noqa: E402
from picklabel.autolabel.run import run  # noqa: E402
from picklabel.eval.autolabel_eval import load_records  # noqa: E402

VARIANTS = {
    "full pipeline": {},
    "no exposure / white-balance gains": {"change.normalize_color": False},
    "no shift tolerance": {"change.shift_px": 0},
    "neither": {"change.normalize_color": False, "change.shift_px": 0},
    "no GrabCut edge snapping": {"segment.grabcut": False},
}


def configure(base: AutoLabelConfig, overrides: dict) -> AutoLabelConfig:
    cfg = copy.deepcopy(base)
    for key, value in overrides.items():
        section, field = key.split(".")
        setattr(getattr(cfg, section), field, value)
    cfg.propagate.enabled = False  # measures single-pick masks only
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--out", default="report/autolabel/ablation.md")
    args = ap.parse_args()
    base = AutoLabelConfig.from_yaml(ROOT / "configs" / "autolabel.yaml")
    rows = ["| Variant | mean IoU | IoU >= 0.75 | false-change px (median, % of frame) |", "|---|---|---|---|"]
    for name, overrides in VARIANTS.items():
        with tempfile.TemporaryDirectory() as tmp:
            run(args.pairs, tmp, AutoLabeler(configure(base, overrides)), previews=False, verbose=False)
            recs = load_records(tmp)
            ious = np.array([r["gold_iou"] for r in recs])
            changed = np.array([r["align"]["change"]["changed_frac"] for r in recs])
        rows.append(f"| {name} | {ious.mean():.3f} | {(ious >= 0.75).mean():.0%} | {100 * np.median(changed):.1f}% |")
        print(rows[-1], flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"Auto-labeler ablation on `{args.pairs}` ({len(recs)} picks with gold masks)\n\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
