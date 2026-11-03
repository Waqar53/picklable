"""Folder layouts in and out: photo import, WS2 mask conversion, a full auto-label run with Roboflow dry-run."""

import json

import cv2
import numpy as np

from picklabel.autolabel.pipeline import AutoLabeler
from picklabel.autolabel.run import run
from picklabel.data.capture import import_photos
from picklabel.data.download import ws2_to_coco
from picklabel.io import find_pick_dirs, load_pair, save_pair, write_json
from picklabel.roboflow_sync.coco import load_coco
from picklabel.roboflow_sync.upload import RoboflowSync


def test_import_photos_pairs_alternating_shots(tmp_path):
    src = tmp_path / "phone"
    src.mkdir()
    for i in range(5):  # before, after, before, after, cleared tray
        cv2.imwrite(str(src / f"IMG_{i:04d}.jpg"), np.full((40, 60, 3), 20 * i, np.uint8))
    dirs = import_photos(src, tmp_path / "pairs", session="s1")
    assert [d.name for d in dirs] == ["000", "001"]
    assert (tmp_path / "pairs" / "s1" / "final_after.jpg").exists()
    meta = json.loads((dirs[0] / "pick.json").read_text())
    assert meta["grasp_px"] is None  # filled in later by `picklabel annotate`


def test_ws2_masks_become_instances(tmp_path):
    root = tmp_path / "test_set"
    (root / "images" / "before" / "v0").mkdir(parents=True)
    (root / "masks" / "before" / "v0").mkdir(parents=True)
    cv2.imwrite(str(root / "images" / "before" / "v0" / "f1.jpg"), np.zeros((50, 80, 3), np.uint8))
    m = np.zeros((50, 80), np.uint8)
    m[5:20, 5:20] = 255
    m[30:45, 50:75] = 255
    cv2.imwrite(str(root / "masks" / "before" / "v0" / "f1.png"), m)
    out = ws2_to_coco(tmp_path, tmp_path / "ws2.json")
    (h, w, insts), = load_coco(out).values()
    assert (h, w) == (50, 80) and len(insts) == 2


def test_autolabel_run_writes_everything_and_dry_runs_roboflow(tmp_path, synth_session):
    pairs, final_after, cleared = synth_session
    for p in pairs[:3]:
        save_pair(p, tmp_path / "pairs")
    write_json(tmp_path / "pairs" / pairs[0].session / "session.json", {"cleared": False})
    sync = RoboflowSync(dry_run=True, ledger_path=tmp_path / "out" / "ledger.jsonl")
    summary = run(tmp_path / "pairs", tmp_path / "out", AutoLabeler(), previews=True, sync=sync, verbose=False)
    assert summary["n_picks"] == 3 and summary["gold"]["n"] == 3
    for sub in ("images", "coco", "previews"):
        assert len(list((tmp_path / "out" / sub).iterdir())) == 3
    calls = [json.loads(line) for line in (tmp_path / "out" / "ledger.jsonl").read_text().splitlines()]
    assert len(calls) == 3 and all(c["dry_run"] for c in calls)
    # frames aren't complete (session not cleared), so nothing goes up as plain ground truth
    assert all(c["kwargs"].get("is_prediction") for c in calls if "annotation_path" in c["kwargs"])
    assert len(find_pick_dirs(tmp_path / "pairs")) == 3 and load_pair(find_pick_dirs(tmp_path / "pairs")[0])
