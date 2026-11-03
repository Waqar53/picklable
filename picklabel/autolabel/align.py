"""Align the after frame onto the before frame.

Two scene models:

* ``static``: the scene is fixed and only the camera moves a little (home tray, stopped belt).
  Coarse-to-fine ECC, with ORB + RANSAC as the fallback.
* ``belt``: the content translates with the conveyor. A tracking robot already knows the
  belt displacement from the encoder, so it is used as the prior and phase correlation
  refines it. Only the overlap of the two views is valid.

Convention: ``H`` maps pixel coordinates in the before frame to the after frame, and
``warped(p) = after(H p)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from picklabel.masks import erode

MOTIONS = {
    "translation": cv2.MOTION_TRANSLATION,
    "euclidean": cv2.MOTION_EUCLIDEAN,
    "affine": cv2.MOTION_AFFINE,
    "homography": cv2.MOTION_HOMOGRAPHY,
}


@dataclass
class AlignConfig:
    mode: str = "static"  # static | belt
    motion: str = "affine"
    work_max_side: int = 960
    max_translation_frac: float = 0.2
    belt_roi: list[int] | None = None  # [x1, y1, x2, y2] in full-resolution pixels
    belt_search_px: float = 24.0
    ecc_iters: int = 100
    ecc_eps: float = 1e-5
    max_residual: float = 14.0  # median grey-level error above which alignment is distrusted


@dataclass
class Alignment:
    warped: np.ndarray
    valid: np.ndarray
    H: np.ndarray
    method: str
    ok: bool
    residual: float

    def info(self) -> dict:
        return {
            "method": self.method,
            "ok": bool(self.ok),
            "residual": round(float(self.residual), 2),
            "tx": round(float(self.H[0, 2]), 2),
            "ty": round(float(self.H[1, 2]), 2),
        }


def _scale_for(shape: tuple[int, ...], max_side: int) -> float:
    return min(1.0, max_side / float(max(shape[:2])))


def _gray(img: np.ndarray, scale: float) -> np.ndarray:
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.GaussianBlur(g, (0, 0), 1.0)


def _to3x3(warp: np.ndarray) -> np.ndarray:
    warp = np.asarray(warp, np.float64)
    return warp if warp.shape == (3, 3) else np.vstack([warp, [0.0, 0.0, 1.0]])


def rescale_h(H: np.ndarray, s: float) -> np.ndarray:
    """Express a transform defined on images scaled by 1 in images scaled by ``s``."""
    S = np.diag([s, s, 1.0])
    return S @ H @ np.linalg.inv(S)


def translation_h(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def _ecc(g_before: np.ndarray, g_after: np.ndarray, cfg: AlignConfig) -> np.ndarray:
    """Coarse-to-fine ECC. Raises cv2.error if even the coarsest level fails."""
    motion = MOTIONS[cfg.motion]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg.ecc_iters, cfg.ecc_eps)
    pyramid = [(g_before, g_after)]
    while len(pyramid) < 3 and min(pyramid[-1][0].shape) > 160:
        b, a = pyramid[-1]
        pyramid.append((cv2.pyrDown(b), cv2.pyrDown(a)))
    H = np.eye(3)
    for level in reversed(range(len(pyramid))):
        b, a = pyramid[level]
        warp = H if motion == cv2.MOTION_HOMOGRAPHY else H[:2]
        try:
            _, warp = cv2.findTransformECC(b, a, warp.astype(np.float32), motion, criteria, None, 5)
            H = _to3x3(warp)
        except cv2.error:
            if level == len(pyramid) - 1:
                raise
        if level > 0:
            # pyrDown halves (rounding up); map the estimate to the next finer level.
            fb = pyramid[level - 1][0].shape[1] / b.shape[1]
            H = rescale_h(H, fb)
    return H


def _orb(g_before: np.ndarray, g_after: np.ndarray, motion: str) -> np.ndarray | None:
    orb = cv2.ORB_create(4000)
    kb, db = orb.detectAndCompute(g_before, None)
    ka, da = orb.detectAndCompute(g_after, None)
    if db is None or da is None or len(kb) < 12 or len(ka) < 12:
        return None
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(db, da, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 12:
        return None
    src = np.float32([kb[m.queryIdx].pt for m in good])
    dst = np.float32([ka[m.trainIdx].pt for m in good])
    if motion == "homography":
        M, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    elif motion == "affine":
        M, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    else:
        M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or inliers is None or int(inliers.sum()) < 10:
        return None
    return _to3x3(M)


def is_plausible(H: np.ndarray, shape: tuple[int, int], max_translation_frac: float) -> bool:
    """Reject warps that a shaking camera or a moving belt could not produce."""
    if not np.all(np.isfinite(H)):
        return False
    det = float(np.linalg.det(H[:2, :2]))
    if not 0.85 < det < 1.18 or abs(H[2, 0]) > 1e-3 or abs(H[2, 1]) > 1e-3:
        return False
    h, w = shape
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float64).T
    moved = H @ corners
    moved = moved[:2] / moved[2]
    shift = np.abs(moved - corners[:2]).max()
    return shift < max_translation_frac * max(h, w)


def _phase_shift(g_before: np.ndarray, g_after: np.ndarray) -> tuple[float, float, float]:
    b, a = g_before.astype(np.float32), g_after.astype(np.float32)
    window = cv2.createHanningWindow((b.shape[1], b.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(b, a, window)
    return float(dx), float(dy), float(response)


def _belt_shift(g_before: np.ndarray, g_after: np.ndarray, prior: tuple[float, float] | None,
                roi: tuple[int, int, int, int] | None, search_px: float) -> tuple[float, float, str]:
    if roi is not None:
        x1, y1, x2, y2 = roi
        g_before, g_after = g_before[y1:y2, x1:x2], g_after[y1:y2, x1:x2]
    if prior is None:
        dx, dy, _ = _phase_shift(g_before, g_after)
        return dx, dy, "belt-phase"
    # Undo the encoder shift (back(p) = after(p + prior)), then phase correlation finds the residual.
    h, w = g_after.shape
    back = cv2.warpAffine(g_after, np.float32([[1, 0, prior[0]], [0, 1, prior[1]]]), (w, h),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    rx, ry, _ = _phase_shift(g_before, back)
    if np.hypot(rx, ry) > search_px:
        rx = ry = 0.0
    return prior[0] + rx, prior[1] + ry, "belt-encoder+phase"


def align_pair(before: np.ndarray, after: np.ndarray, cfg: AlignConfig | None = None,
               belt_shift_px: tuple[float, float] | None = None) -> Alignment:
    cfg = cfg or AlignConfig()
    h, w = before.shape[:2]
    if after.shape[:2] != (h, w):
        after = cv2.resize(after, (w, h), interpolation=cv2.INTER_AREA)
    s = _scale_for(before.shape, cfg.work_max_side)
    gb, ga = _gray(before, s), _gray(after, s)
    work_shape = gb.shape

    ok = True
    if cfg.mode == "belt":
        prior = (belt_shift_px[0] * s, belt_shift_px[1] * s) if belt_shift_px is not None else None
        roi = tuple(int(round(v * s)) for v in cfg.belt_roi) if cfg.belt_roi else None
        dx, dy, method = _belt_shift(gb, ga, prior, roi, cfg.belt_search_px * s)
        Hs = translation_h(dx, dy)
    elif cfg.mode == "static":
        Hs, method = None, f"ecc-{cfg.motion}"
        try:
            Hs = _ecc(gb, ga, cfg)
        except cv2.error:
            Hs = None
        if Hs is None or not is_plausible(Hs, work_shape, cfg.max_translation_frac):
            Hs, method = _orb(gb, ga, cfg.motion), f"orb-{cfg.motion}"
        if Hs is None or not is_plausible(Hs, work_shape, cfg.max_translation_frac):
            Hs, method, ok = np.eye(3), "identity", False
    else:
        raise ValueError(f"unknown align mode: {cfg.mode!r}")

    H = rescale_h(Hs, 1.0 / s)
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    warped = cv2.warpPerspective(after, H, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.warpPerspective(np.ones((h, w), np.uint8), H, (w, h),
                                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0) > 0
    valid = erode(valid, 2)

    ga_w = cv2.warpPerspective(ga, Hs, (work_shape[1], work_shape[0]), flags=flags)
    valid_s = cv2.resize(valid.astype(np.uint8), (work_shape[1], work_shape[0]),
                         interpolation=cv2.INTER_NEAREST) > 0
    diff = np.abs(gb.astype(np.int16) - ga_w.astype(np.int16))[valid_s]
    residual = float(np.median(diff)) if diff.size else 255.0
    if residual > cfg.max_residual or valid.mean() < 0.3:
        ok = False
    return Alignment(warped=warped, valid=valid, H=H, method=method, ok=ok, residual=residual)
