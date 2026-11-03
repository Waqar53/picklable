"""Assemble the static demo site (GitHub Pages): page + the picklabel modules the browser runs + examples.

    python scripts/build_site.py --out build/site
    python -m http.server 8000 --directory build/site

The browser runs the same Python files as this repo (Pyodide). Example results are also
precomputed here with that code, so the page shows them instantly while Python loads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "site")]

from picklabel.io import find_pick_dirs, read_json  # noqa: E402

PY_FILES = [
    "picklabel/__init__.py", "picklabel/types.py", "picklabel/masks.py", "picklabel/viz.py",
    "picklabel/autolabel/__init__.py", "picklabel/autolabel/align.py", "picklabel/autolabel/change.py",
    "picklabel/autolabel/pipeline.py", "picklabel/autolabel/score.py", "picklabel/autolabel/segment.py",
    "picklabel/autolabel/selector.py", "picklabel/roboflow_sync/__init__.py", "picklabel/roboflow_sync/upload.py",
]
PAGE_FILES = ["index.html", "style.css", "app.js", "worker.js"]
ASSETS = {
    "report/synthetic/headline_synthetic.png": "headline_synthetic.png",
    "report/synthetic/pr_synthetic.png": "pr_synthetic.png",
    "report/autolabel/sam2_tiny/calibration.png": "calibration.png",
    "docs/img/pick_accept.jpg": "pick_accept.jpg",
    "docs/img/picklabel.gif": "picklabel.gif",
}


def describe(meta: dict) -> str:
    if meta.get("missed"):
        return "gripper missed: nothing removed"
    kind = "clear film" if meta.get("film") else ("dark plastic" if meta.get("dark") else "plastic")
    return kind + (", gripper confirmed" if meta.get("grasp_success") else "")


def build(out: Path, examples: Path) -> None:
    import demo_bridge

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in PAGE_FILES:
        shutil.copy2(ROOT / "site" / f, out / f)
    page = (out / "index.html").read_text()
    for f in ("style.css", "app.js"):  # cache-busting: a redeploy must never serve stale CSS / JS
        digest = hashlib.sha1((out / f).read_bytes()).hexdigest()[:8]
        page = page.replace(f'"{f}"', f'"{f}?v={digest}"')
    (out / "index.html").write_text(page)
    for f in PY_FILES:
        (out / "py" / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, out / "py" / f)
    shutil.copy2(ROOT / "site" / "demo_bridge.py", out / "py" / "demo_bridge.py")
    (out / "py" / "manifest.json").write_text(json.dumps([*PY_FILES, "demo_bridge.py"]))

    entries = []
    for n, d in enumerate(find_pick_dirs(examples), start=1):
        meta = read_json(d / "pick.json")
        key = f"{d.parent.name}-{d.name}"
        dst = out / "examples" / key
        dst.mkdir(parents=True)
        for name in ("before.jpg", "after.jpg", "gt_mask.png"):
            shutil.copy2(d / name, dst / name)
        demo_bridge.load(key, (d / "before.jpg").read_bytes(), (d / "after.jpg").read_bytes(),
                         (d / "gt_mask.png").read_bytes())
        result = demo_bridge.run(key, *meta["grasp_px"], meta.get("grasp_success"))
        (dst / "result.json").write_text(result)
        r = json.loads(result)
        print(f"{key}: {r['route']} q={r['score']} iou={r['iou']}")
        entries.append({
            "key": key, "label": f"Pick {n}" + (" · missed" if meta.get("missed") else ""), "note": describe(meta),
            "before": f"examples/{key}/before.jpg", "after": f"examples/{key}/after.jpg",
            "gt": f"examples/{key}/gt_mask.png", "result": f"examples/{key}/result.json",
            "grasp": meta["grasp_px"], "grasp_success": meta.get("grasp_success"),
        })
    (out / "examples" / "examples.json").write_text(json.dumps(entries, indent=1))

    (out / "assets").mkdir()
    for src, name in ASSETS.items():
        shutil.copy2(ROOT / src, out / "assets" / name)
    (out / ".nojekyll").write_text("")  # GitHub Pages must serve __init__.py (Jekyll hides _files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/site")
    ap.add_argument("--examples", default=str(ROOT / "examples" / "synthetic"))
    args = ap.parse_args()
    build(Path(args.out), Path(args.examples))
    print(f"site -> {args.out}")


if __name__ == "__main__":
    main()
