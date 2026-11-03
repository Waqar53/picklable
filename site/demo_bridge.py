"""Bridge between the demo page (a Web Worker) and the picklabel package running in Pyodide.

Everything returned to JavaScript is a JSON string, images included (JPEG data URLs).
"""

import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np

from picklabel.autolabel.pipeline import AutoLabeler
from picklabel.masks import mask_iou
from picklabel.roboflow_sync.upload import RoboflowSync, UploadItem, make_tags
from picklabel.types import PickPair
from picklabel.viz import ROUTE_BGR, draw_point, heatmap, overlay

MAX_SIDE = 1280  # uploads from a phone are downscaled to keep the browser responsive
LABELER = AutoLabeler()  # same defaults as configs/autolabel.yaml
SYNC = RoboflowSync(dry_run=True, ledger_path="/tmp/picklabel_demo_ledger.jsonl")
_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}


def _bytes(buf) -> bytes:
    return buf.to_bytes() if hasattr(buf, "to_bytes") else bytes(buf)


def _decode(buf, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(_bytes(buf), np.uint8), flags)
    if img is None:
        raise ValueError("could not decode image")
    return img


def _jpeg(img: np.ndarray, quality: int = 88) -> str:
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode()


def load(key: str, before, after, gt=None) -> str:
    b, a = _decode(before), _decode(after)
    s = min(1.0, MAX_SIDE / max(b.shape[:2]))
    if s < 1.0:
        b = cv2.resize(b, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if a.shape != b.shape:
        a = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)
    g = _decode(gt, cv2.IMREAD_GRAYSCALE) > 127 if gt is not None else None
    _cache[key] = (b, a, g)
    return json.dumps({"width": int(b.shape[1]), "height": int(b.shape[0]), "scale": s})


def run(key: str, gx: float, gy: float, grasp_success=None) -> str:
    b, a, g = _cache[key]
    pair = PickPair(pick_id=key, before=b, after=a, grasp_xy=(float(gx), float(gy)), session="demo",
                    grasp_success=grasp_success)
    t = time.perf_counter()
    label = LABELER.label(pair, keep_debug=True)
    ms = (time.perf_counter() - t) * 1000

    color = ROUTE_BGR[label.route]
    change = heatmap(label.change_score) if label.change_score is not None else np.zeros_like(b)
    if label.region is not None:
        cs, _ = cv2.findContours(label.region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(change, cs, -1, (255, 255, 255), max(1, b.shape[1] // 320), cv2.LINE_AA)
    result = draw_point(overlay(b, label.mask, color), label.grasp_xy)
    after = draw_point(label.after_aligned if label.after_aligned is not None else a, label.grasp_xy)

    iou = None  # only meaningful for a mask, made with the grasp on the item that was picked
    gxi, gyi = int(round(label.grasp_xy[0])), int(round(label.grasp_xy[1]))
    if g is not None and label.mask is not None and g[gyi, gxi]:
        iou = mask_iou(label.mask, g)

    item = UploadItem(key=f"demo__{key}", image_path=Path(f"{key}.jpg"), route=label.route,
                      coco_path=Path(f"{key}.json"), n_instances=int(label.mask is not None),
                      tags=make_tags(label.route, label.source, label.score, "demo", label.reasons),
                      metadata={"quality": round(label.score, 3), "route": label.route.value,
                                "source": label.source})
    signals = {k: round(float(label.signals[k]), 3)
               for k in ("agreement", "core", "support", "contrast", "shape", "source") if k in label.signals}
    return json.dumps({
        "route": label.route.value, "score": round(float(label.score), 3), "source": label.source,
        "reasons": label.reasons, "signals": signals, "ms": round(ms), "iou": iou,
        "timings": {k: round(float(v)) for k, v in label.timings_ms.items()},
        "align": {k: v for k, v in label.align.items() if k != "change"},
        "gains": label.align.get("change", {}).get("gains_bgr"),
        "mask_area": int(label.mask.sum()) if label.mask is not None else 0,
        "images": {"after": _jpeg(after), "change": _jpeg(change), "result": _jpeg(result)},
        "payload": SYNC.call_args(item, "picklabel-demo"),
    })
