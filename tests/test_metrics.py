import cv2
import numpy as np

from picklabel.eval.metrics import average_precision, evaluate, match_image, pr_curve, recall_at_precision


def disk(c, r, shape=(100, 100)):
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, c, r, 1, -1)
    return m.astype(bool)


def test_perfect_predictions():
    gts = [[disk((30, 30), 10), disk((70, 70), 12)]]
    rep, curve = evaluate([(gts[0], [0.9, 0.8])], gts)
    assert rep.ap50 > 0.99 and rep.recall_at_p90 == 1.0
    assert rep.at_threshold["false_positives"] == 0


def test_duplicates_and_false_positives_are_counted():
    g = disk((30, 30), 10)
    preds = [g, np.roll(g, 1, axis=1), disk((80, 20), 6)]
    ev = match_image(preds, [0.9, 0.8, 0.7], [g])
    assert ev.tp.tolist() == [True, False, False]
    assert ev.duplicate.tolist() == [False, True, False]
    curve = pr_curve([ev])
    assert curve.precision.round(3).tolist() == [1.0, 0.5, 0.333]
    r, t = recall_at_precision(curve, 0.9)
    assert r == 1.0 and t == 0.9


def test_active_learning_ranks_frames_near_the_threshold_first():
    from picklabel.eval.active import frame_uncertainty, rank_frames

    assert frame_uncertainty([0.42], 0.42) == 1.0
    assert frame_uncertainty([0.95, 0.05], 0.42) == 0.0
    recs = [{"image": "sure.jpg", "scores": [0.97, 0.02]}, {"image": "unsure.jpg", "scores": [0.45, 0.38]}]
    assert [r["image"] for r in rank_frames(recs, 0.42)] == ["unsure.jpg", "sure.jpg"]


def test_ap_is_zero_without_gt_matches():
    g = disk((30, 30), 10)
    ev = match_image([disk((80, 80), 10)], [0.9], [g])
    assert average_precision(pr_curve([ev])) == 0.0
