"""From an instance mask to a 3D pick point the arm can use.

The most interior pixel of the mask (distance-transform peak) is the safest place to close
a gripper on a deformable bag. Its depth is the median of the valid depth around it, so a
missing or noisy stereo pixel doesn't move the point. It is back-projected with the pinhole
intrinsics. The minimum-area rectangle gives a gripper yaw, and the distance-transform peak
gives a width proxy.

This is a hand-off format, not a grasp planner: Grip's gripper and planner (cuRobo) decide
the actual grasp. With the ZED, ``depth_m`` is the registered depth map (32-bit float,
metres) and the intrinsics come from ``camera_info``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PickPoint:
    xyz_m: tuple[float, float, float]  # camera frame: x right, y down, z forward
    pixel: tuple[int, int]
    yaw_rad: float  # long axis of the item, in the image plane
    width_px: float  # 2 x distance from the pick pixel to the nearest edge
    depth_valid_frac: float


def pick_point_3d(mask: np.ndarray, depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                  window: int = 7) -> PickPoint | None:
    if mask.sum() == 0:
        return None
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    v, u = np.unravel_index(int(np.argmax(dist)), dist.shape)
    r = window // 2
    win = depth_m[max(v - r, 0): v + r + 1, max(u - r, 0): u + r + 1]
    good = win[np.isfinite(win) & (win > 0)]
    if good.size == 0:  # fall back to any valid depth on the item
        vals = depth_m[mask & np.isfinite(depth_m) & (depth_m > 0)]
        if vals.size == 0:
            return None
        z = float(np.median(vals))
        valid = vals.size / mask.sum()
    else:
        z = float(np.median(good))
        valid = good.size / win.size
    x, y = (u - cx) * z / fx, (v - cy) * z / fy
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (_, _), (w, h), angle = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    yaw = np.deg2rad(angle if w >= h else angle + 90.0)
    return PickPoint((float(x), float(y), z), (int(u), int(v)), float(yaw), float(2 * dist.max()), float(valid))
