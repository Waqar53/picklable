import cv2
import numpy as np

from picklabel.masks import (
    iou_matrix,
    mask_bbox,
    mask_iou,
    mask_nms,
    mask_to_polygons,
    polygons_to_mask,
    rle_decode,
    rle_encode,
)


def disk_mask(h, w, c, r):
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, c, r, 1, -1)
    return m.astype(bool)


def test_rle_roundtrip_matches_column_major_counts():
    m = np.zeros((4, 3), bool)
    m[1:3, 1] = True
    m[0, 0] = True
    rle = rle_encode(m)
    assert rle["size"] == [4, 3]
    assert rle["counts"][0] == 0  # starts with a foreground pixel
    assert np.array_equal(rle_decode(rle), m)
    rnd = np.random.default_rng(0).random((37, 53)) > 0.6
    assert np.array_equal(rle_decode(rle_encode(rnd)), rnd)


def test_polygon_roundtrip_keeps_shape():
    m = disk_mask(120, 160, (80, 60), 30)
    back = polygons_to_mask(mask_to_polygons(m), 120, 160)
    assert mask_iou(m, back) > 0.95


def test_bbox_iou_and_matrix():
    a = disk_mask(100, 100, (30, 30), 10)
    b = disk_mask(100, 100, (70, 70), 10)
    assert mask_bbox(a) == (20, 20, 41, 41)
    assert mask_bbox(np.zeros((5, 5), bool)) is None
    m = iou_matrix([a, b], [b, a])
    assert m[0, 1] == 1.0 and m[1, 0] == 1.0 and m[0, 0] == 0.0


def test_mask_nms_drops_duplicates_keeps_best():
    a = disk_mask(100, 100, (50, 50), 20)
    a2 = disk_mask(100, 100, (51, 50), 20)
    b = disk_mask(100, 100, (15, 15), 8)
    keep = mask_nms([a, a2, b], [0.5, 0.9, 0.3], 0.6)
    assert keep == [1, 2]
