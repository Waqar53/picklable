import cv2
import numpy as np
from conftest import textured

from picklabel.autolabel.align import AlignConfig, align_pair
from picklabel.autolabel.change import ChangeConfig, compute_change
from picklabel.masks import mask_iou


def warp(img, A):
    return cv2.warpAffine(img, A, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def test_static_alignment_recovers_camera_shake():
    before = textured(seed=1)
    ang, tx, ty = np.deg2rad(0.6), 3.5, -2.0
    A = np.array([[np.cos(ang), -np.sin(ang), tx], [np.sin(ang), np.cos(ang), ty]])
    after = warp(before, A)  # after(A p) = before(p): H should equal A
    al = align_pair(before, after, AlignConfig(motion="affine"))
    assert al.ok
    corners = np.array([[0, 0, 1], [320, 0, 1], [0, 240, 1], [320, 240, 1]], float).T
    assert np.abs(al.H[:2] @ corners - A @ corners).max() < 0.3


def test_belt_alignment_uses_encoder_prior_and_refines_it():
    rng = np.random.default_rng(3)
    wide = cv2.GaussianBlur(rng.normal(0, 1, (240, 520)).astype(np.float32), (0, 0), 3)
    wide = ((wide - wide.min()) / np.ptp(wide) * 200 + 30).astype(np.uint8)
    wide = cv2.cvtColor(wide, cv2.COLOR_GRAY2BGR)
    before, after = wide[:, 100:420], wide[:, 40:360]  # content moved +60 px in x
    cfg = AlignConfig(mode="belt")
    al = align_pair(before, after, cfg, belt_shift_px=(57.0, 1.0))  # encoder off by a few px
    assert abs(al.H[0, 2] - 60) < 0.6 and abs(al.H[1, 2]) < 0.6
    assert al.valid[:, :250].any() and not al.valid[:, -40:].any()  # only the overlap is valid
    al2 = align_pair(before, after, cfg)  # no encoder: phase correlation alone
    assert abs(al2.H[0, 2] - 60) < 0.6


def test_no_false_change_under_exposure_and_white_balance_drift():
    before = textured(seed=2)
    after = np.clip(before.astype(np.float32) * [1.08, 0.95, 1.12], 0, 255).astype(np.uint8)
    ch = compute_change(before, after, None, ChangeConfig())
    assert ch.mask.mean() < 0.001


def test_removed_item_is_found_and_nudged_neighbour_is_not():
    before = textured(seed=4)
    item = np.zeros(before.shape[:2], np.uint8)
    cv2.ellipse(item, (100, 120), (40, 25), 15, 0, 360, 1, -1)
    item = item.astype(bool)
    neighbour = np.zeros_like(item)
    cv2.circle(neighbour.view(np.uint8), (220, 120), 30, 1, -1)
    scene = before.copy()
    scene[item] = (30, 160, 220)
    tex = textured(seed=9)
    scene[neighbour] = tex[neighbour]
    after = before.copy()
    shifted = np.roll(neighbour, (2, 3), axis=(0, 1))
    after[shifted] = np.roll(tex, (2, 3), axis=(0, 1))[shifted]  # neighbour nudged 3 px, item gone
    ch = compute_change(scene, after, None, ChangeConfig())
    assert mask_iou(ch.mask & item, item) > 0.85
    assert (ch.mask & neighbour & ~item).sum() < 0.1 * neighbour.sum()
