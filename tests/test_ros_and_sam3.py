"""ROS-free parts of the detector node, and the SAM 3 sweep on a fake candidate cache."""

from types import SimpleNamespace

import cv2
import numpy as np
from picklabel_ros.inference import parse_class_thresholds, parse_prompt_text, results_to_detections

from picklabel.cli import COMMANDS
from picklabel.eval.sam3_baseline import sweep


class T:  # minimal stand-in for a torch tensor
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def __len__(self):
        return len(self.a)


def disk(c, r, shape=(80, 100)):
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, c, r, 1, -1)
    return m.astype(np.float32)


def fake_result():
    return SimpleNamespace(
        masks=SimpleNamespace(data=T([disk((30, 30), 10), disk((70, 40), 12), disk((50, 60), 5)])),
        boxes=SimpleNamespace(conf=T([0.9, 0.6, 0.3]), cls=T([0, 1, 0])),
        names={0: "plastic", 1: "film"},
    )


def test_prompt_text_parsing_matches_sam3_ros():
    assert parse_prompt_text("['plastic bag', 'film']") == ["plastic bag", "film"]
    assert parse_prompt_text("plastic, film") == ["plastic", "film"]
    assert parse_prompt_text([""]) == []
    assert parse_class_thresholds(["plastic=0.45", "", "film=0.3"]) == {"plastic": 0.45, "film": 0.3}


def test_detections_follow_sam3_ros_conventions():
    dets = results_to_detections(fake_result(), (80, 100), threshold=0.5)
    assert [d.instance_id for d in dets] == ["plastic_0", "film_1"]
    x, y, w, h = dets[0].bbox_xywh
    assert (x, y, w, h) == (20, 20, 21, 21)  # bounding rect of the mask pixels, like sam3_ros
    assert 0.0 <= dets[0].score <= 1.0


def test_class_filter_and_per_class_thresholds():
    assert [d.class_name for d in results_to_detections(fake_result(), (80, 100), 0.5, ["film"])] == ["film"]
    dets = results_to_detections(fake_result(), (80, 100), 0.5, class_thresholds={"plastic": 0.25})
    assert sorted(d.score for d in dets) == [0.3, 0.6, 0.9]


def test_sam3_sweep_counts_cross_prompt_duplicates():
    g = disk((30, 30), 10).astype(bool)
    rec = {"image": "a.jpg", "masks": [g, g, disk((80, 60), 6).astype(bool)], "scores": [0.9, 0.85, 0.8],
           "labels": ["plastic bag", "plastic film", "plastic film"], "set_image_ms": 100.0,
           "prompt_ms": {"plastic bag": 20.0, "plastic film": 20.0}}
    rows, _ = sweep([rec], {"a.jpg": [g]}, {"B": ["plastic bag", "plastic film"]})
    node = next(r for r in rows if r["mode"] == "node")
    nms = next(r for r in rows if r["mode"] == "nms")
    assert node["at_0.75"]["duplicates"] == 1 and nms["at_0.75"]["duplicates"] == 0
    assert node["at_0.75"]["precision"] < nms["at_0.75"]["precision"]
    assert node["ms_per_frame_p50"] == 140.0


def test_cli_lists_every_stage():
    for cmd in ("autolabel", "build-dataset", "train", "sam3-baseline", "eval-student", "export"):
        assert cmd in COMMANDS
