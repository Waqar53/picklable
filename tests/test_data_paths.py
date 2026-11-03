"""COCO export, dataset building (no session leaks), Roboflow call mapping, pick point."""

import json

import cv2
import numpy as np

from picklabel.deploy.pick_point import pick_point_3d
from picklabel.io import write_image
from picklabel.masks import mask_iou
from picklabel.roboflow_sync.coco import CocoBuilder, load_coco
from picklabel.roboflow_sync.upload import RoboflowSync, UploadItem, make_tags
from picklabel.train.build_dataset import build
from picklabel.types import Route


def blob(c, r, shape=(120, 160)):
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, c, r, 1, -1)
    return m.astype(bool)


def fake_run(tmp_path, sessions=4, per_session=3, route="accept"):
    run = tmp_path / "run"
    coco = CocoBuilder(["plastic"])
    img = np.full((120, 160, 3), 60, np.uint8)
    for s in range(sessions):
        for k in range(per_session):
            name = f"s{s}__{k:03d}.jpg"
            write_image(run / "images" / name, img)
            i = coco.add_image(name, 120, 160, {"session": f"s{s}", "complete": True, "route": route})
            coco.add_instance(i, blob((40 + 10 * k, 60), 15), "plastic", {"route": route})
    coco.save(run / "annotations.json")
    return run


def test_coco_roundtrip(tmp_path):
    m = blob((80, 60), 25)
    b = CocoBuilder(["plastic"])
    i = b.add_image("a.jpg", 120, 160)
    b.add_instance(i, m, "plastic", {"route": "accept", "score": 0.9})
    b.save(tmp_path / "c.json")
    (h, w, insts), = load_coco(tmp_path / "c.json").values()
    assert (h, w) == (120, 160) and mask_iou(insts[0][0], m) > 0.95
    assert insts[0][2]["picklabel"]["route"] == "accept"


def test_dataset_split_is_by_session(tmp_path):
    manifest = build([fake_run(tmp_path)], tmp_path / "ds", {"accept"}, val_frac=0.25, seed=0)
    splits = manifest["sessions"]
    assert set(splits.values()) == {"train", "val"}
    for split in ("train", "val"):
        stems = {p.stem.split("__")[0] for p in (tmp_path / "ds" / "images" / split).iterdir()}
        assert all(splits[s] == split for s in stems)  # every image of a session is in one split
    line = next((tmp_path / "ds" / "labels" / "train").glob("*.txt")).read_text().split()
    assert line[0] == "0" and all(0.0 <= float(v) <= 1.0 for v in line[1:])


def test_review_frames_excluded_from_accept_only_dataset(tmp_path):
    manifest = build([fake_run(tmp_path, route="review")], tmp_path / "ds", {"accept"})
    assert manifest["skipped"]["route"] == 12 and not manifest["images"]


def test_roboflow_call_args_follow_route(tmp_path):
    sync = RoboflowSync(dry_run=True, ledger_path=tmp_path / "ledger.jsonl")
    tags = make_tags(Route.REVIEW, "sam2", 0.55, "s1", ["weak_change"])
    assert {"route:review", "source:sam2", "q:0.5", "session:s1", "reason:weak_change"} <= set(tags)
    accept = sync.call_args(UploadItem("a", tmp_path / "a.jpg", Route.ACCEPT, tmp_path / "a.json", 1), "b")
    review = sync.call_args(UploadItem("r", tmp_path / "r.jpg", Route.REVIEW, tmp_path / "r.json", 1), "b")
    hard = sync.call_args(UploadItem("h", tmp_path / "h.jpg", Route.HARD, None, 0), "b")
    assert "is_prediction" not in accept and accept["annotation_path"].endswith("a.json")
    assert review["is_prediction"] is True and review["add_to_dataset"] is False
    assert "annotation_path" not in hard
    rec = sync.push(UploadItem("a", tmp_path / "a.jpg", Route.ACCEPT, tmp_path / "a.json", 1), "b")
    assert rec["dry_run"] and json.loads((tmp_path / "ledger.jsonl").read_text())["key"] == "a"


def test_pick_point_on_a_flat_tray():
    mask = blob((80, 60), 20)
    depth = np.full((120, 160), 0.55, np.float32)
    depth[60, 80] = np.nan  # a hole in the stereo depth must not matter
    pp = pick_point_3d(mask, depth, fx=500, fy=500, cx=80, cy=60)
    assert pp is not None and abs(pp.xyz_m[2] - 0.55) < 1e-6
    assert abs(pp.xyz_m[0]) < 0.003 and abs(pp.xyz_m[1]) < 0.003
