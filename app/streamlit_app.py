"""PickLabel demo: auto-label one pick, run the student, see the results.

    streamlit run app/streamlit_app.py

Runs on CPU (change map + GrabCut by default). Point it at SAM 2.1 weights for
grasp-prompted masks. The examples are synthetic trays that ship with the repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from picklabel.autolabel.pipeline import AutoLabelConfig, AutoLabeler  # noqa: E402
from picklabel.io import find_pick_dirs, load_pair  # noqa: E402
from picklabel.masks import mask_iou  # noqa: E402
from picklabel.roboflow_sync.upload import RoboflowSync, UploadItem, make_tags  # noqa: E402
from picklabel.types import PickPair  # noqa: E402
from picklabel.viz import autolabel_panel, draw_point  # noqa: E402

st.set_page_config(page_title="PickLabel", layout="wide")


@st.cache_resource
def labeler(sam2_path: str) -> AutoLabeler:
    cfg = AutoLabelConfig.from_yaml(ROOT / "configs" / "autolabel.yaml")
    point = None
    if sam2_path:
        from picklabel.autolabel.segment import Sam2PointSegmenter

        point = Sam2PointSegmenter(sam2_path)
    return AutoLabeler(cfg, point_segmenter=point)


@st.cache_resource
def student(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


def rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode(upload) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)


st.title("PickLabel")
st.caption("Every robot pick is a free label: before/after frames + the grasp point -> a scored instance mask, "
           "routed to Roboflow as ground truth, as a prediction for review, or as a hard example.")

tab_pick, tab_student, tab_results = st.tabs(["Auto-label a pick", "Student detector", "Results"])

with tab_pick:
    left, right = st.columns([1, 3])
    with left:
        examples = sorted(find_pick_dirs(ROOT / "examples")) if (ROOT / "examples").exists() else []
        source = st.radio("Pair", ["Example", "Upload"], horizontal=True, disabled=not examples)
        sam2_path = st.text_input("SAM 2.1 weights (optional)", "", help="e.g. sam2.1_t.pt; empty = change map + GrabCut")
        if source == "Example" and examples:
            choice = st.selectbox("Example pick", examples, format_func=lambda p: f"{p.parent.name}/{p.name}")
            pair = load_pair(choice)
        else:
            b, a = st.file_uploader("Before", ["jpg", "jpeg", "png"]), st.file_uploader("After", ["jpg", "jpeg", "png"])
            if not (b and a):
                st.info("Upload a before and an after frame of the same scene.")
                st.stop()
            before, after = decode(b), decode(a)
            h, w = before.shape[:2]
            pair = PickPair("upload", before, after, (w / 2, h / 2), session="upload")
        h, w = pair.shape
        gx = st.slider("Grasp x", 0, w - 1, int(pair.grasp_xy[0]))
        gy = st.slider("Grasp y", 0, h - 1, int(pair.grasp_xy[1]))
        pair.grasp_xy = (float(gx), float(gy))
        st.image(rgb(draw_point(pair.before, pair.grasp_xy)), caption="before + grasp point", width="stretch")
    with right:
        with st.spinner("Aligning, differencing, segmenting..."):
            label = labeler(sam2_path.strip()).label(pair, keep_debug=True)
        st.image(rgb(autolabel_panel(pair, label, height=360)), width="stretch")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Route", label.route.value.upper())
        c2.metric("Quality score", f"{label.score:.2f}")
        c3.metric("Mask source", label.source)
        c4.metric("Time", f"{label.timings_ms.get('total', 0):.0f} ms")
        if pair.gt_mask is not None and label.mask is not None:
            st.success(f"IoU with the gold mask: {mask_iou(label.mask, pair.gt_mask):.3f}")
        if label.reasons:
            st.warning("Reasons: " + ", ".join(label.reasons))
        sig = {k: round(v, 3) for k, v in label.signals.items()
               if k in ("agreement", "core", "support", "contrast", "shape", "source", "grasp")}
        st.bar_chart(sig, horizontal=True, height=220)
        sync = RoboflowSync(dry_run=True, ledger_path=Path("picklabel_demo_ledger.jsonl"))  # call_args never writes
        item = UploadItem(f"{pair.session}__{pair.pick_id}", Path("before.jpg"), label.route, Path("ann.json"),
                          int(label.mask is not None),
                          make_tags(label.route, label.source, label.score, pair.session, label.reasons))
        st.markdown("**What would be sent to Roboflow** (`Project.single_upload` keyword arguments)")
        st.json(sync.call_args(item, "picklabel-demo"))

with tab_student:
    weights = st.text_input("Student weights (.pt / .onnx)", str(ROOT / "weights" / "student.pt"))
    upload = st.file_uploader("Image", ["jpg", "jpeg", "png"], key="student_img")
    thr = st.slider("Threshold", 0.05, 0.95, 0.5, 0.05)
    if upload and Path(weights).exists():
        img = decode(upload)
        r = student(weights).predict(img, conf=thr, retina_masks=True, verbose=False)[0]
        st.image(rgb(r.plot()), width="stretch")
        n = 0 if r.boxes is None else len(r.boxes)
        st.write(f"{n} detections at threshold {thr:.2f}")
    elif upload:
        st.error(f"weights not found: {weights}")

with tab_results:
    figs = sorted((ROOT / "report").rglob("*.png")) if (ROOT / "report").exists() else []
    if not figs:
        st.info("Run the evaluation scripts to populate report/.")
    for f in figs:
        st.image(str(f), caption=str(f.relative_to(ROOT)), width="stretch")
    results = ROOT / "report" / "results.md"
    if results.exists():
        st.markdown(results.read_text())
