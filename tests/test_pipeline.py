import numpy as np

from picklabel.autolabel.pipeline import AutoLabelConfig, AutoLabeler, is_failed_pick
from picklabel.autolabel.propagate import propagate_session
from picklabel.autolabel.score import ScoreConfig, decide
from picklabel.masks import instances_from_id_map, iou_matrix, mask_iou
from picklabel.types import PickPair, Route


def test_autolabeler_recovers_synthetic_picks(synth_session):
    pairs, _, _ = synth_session
    labeler = AutoLabeler(AutoLabelConfig())
    ious = []
    for p in pairs:
        lb = labeler.label(p)
        if p.meta["missed"]:
            assert is_failed_pick(lb) and lb.route == Route.HARD
            continue
        assert lb.mask is not None
        ious.append(mask_iou(lb.mask, p.gt_mask))
    assert np.mean(ious) > 0.8, ious


def test_failed_pick_routes_to_hard_with_gripper_reason():
    rng = np.random.default_rng(0)
    img = (rng.random((200, 260, 3)) * 255).astype(np.uint8)
    pair = PickPair("x", img, img.copy(), (100, 100), grasp_success=False)
    lb = AutoLabeler().label(pair)
    assert lb.route == Route.HARD and lb.mask is None
    assert lb.reasons == ["no_change_at_grasp", "gripper_empty"]


def test_routing_rules():
    cfg = ScoreConfig()
    good = {"grasp": 1.0, "support": 0.9, "agreement": 0.9, "shape": 1.0}
    assert decide(good, 0.9, cfg, True, True, True)[0] == Route.ACCEPT
    assert decide(good, 0.5, cfg, True, True, None)[0] == Route.REVIEW
    assert decide(good, 0.2, cfg, True, True, None)[0] == Route.HARD
    # gripper says it held nothing although the scene changed: pushed, not picked -> never auto-accepted
    route, score, reasons = decide(good, 0.95, cfg, True, True, False)
    assert route == Route.REVIEW and "gripper_empty_but_scene_changed" in reasons
    # grasp point outside the mask -> cannot be accepted or reviewed
    assert decide({**good, "grasp": 0.0}, 0.95, cfg, True, True, None)[0] == Route.HARD
    # suspect alignment lowers the score
    assert decide(good, 0.8, cfg, True, False, None)[1] < 0.8


def test_propagation_completes_earlier_frames(synth_session):
    pairs, _, cleared = synth_session
    cfg = AutoLabelConfig()
    labeler = AutoLabeler(cfg)
    labels = [labeler.label(p) for p in pairs]
    frames = propagate_session(pairs, labels, cfg.propagate, cfg.align, cfg.change, cleared)
    first = frames[0]
    gts = instances_from_id_map(pairs[0].gt_instances)
    assert len(first.instances) >= 2  # its own item plus items picked later
    ious = iou_matrix([i.mask for i in first.instances], gts)
    assert (ious.max(axis=1) >= 0.5).mean() >= 0.75  # propagated masks land on real items
    assert any(i.propagated for i in first.instances)
