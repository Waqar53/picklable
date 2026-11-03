"""The whole PickLabel loop on synthetic data, end to end, on a laptop.

synthetic sessions -> auto-label -> datasets (R1 manual-equivalent GT vs R2 PickLabel) ->
train two YOLO-seg students -> evaluate both on held-out sessions -> report figures.

This checks that every stage works and that auto-labels are usable as training data. The
numbers describe synthetic trays, not real waste; report them as a smoke test only.

    python scripts/synthetic_e2e.py --work runs/synthetic_e2e --report report/synthetic
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from picklabel.autolabel.pipeline import AutoLabelConfig  # noqa: E402
from picklabel.autolabel.run import build_labeler, run  # noqa: E402
from picklabel.data.synth import SynthConfig, generate  # noqa: E402
from picklabel.eval.data import load_gt  # noqa: E402
from picklabel.eval.eval_student import benchmark, evaluate_records, predict_yolo  # noqa: E402
from picklabel.eval.plots import pr_figure  # noqa: E402
from picklabel.io import read_json, write_json  # noqa: E402
from picklabel.train.build_dataset import build, export_gt_run  # noqa: E402
from picklabel.train.train import train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="runs/synthetic_e2e")
    ap.add_argument("--report", default="report/synthetic")
    ap.add_argument("--train-sessions", type=int, default=30)
    ap.add_argument("--test-sessions", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--sam2", default=None, help="SAM 2.1 weights for the auto-labeler (default: change-only)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume", action="store_true", help="reuse finished stages found in --work")
    args = ap.parse_args()

    work, report_dir = Path(args.work), Path(args.report)
    t0 = time.time()
    log = {}

    print("1/6 synthetic sessions")
    if not (args.resume and (work / "pairs_test").exists()):
        generate(work / "pairs_train", args.train_sessions, seed=1, cfg=SynthConfig(), prefix="tray")
        generate(work / "pairs_test", args.test_sessions, seed=99, cfg=SynthConfig(), prefix="test")

    print("2/6 auto-label the training sessions")
    if args.resume and (work / "autolabel" / "summary.json").exists():
        summary = read_json(work / "autolabel" / "summary.json")
    else:
        labeler = build_labeler(AutoLabelConfig.from_yaml(ROOT / "configs" / "autolabel.yaml"), sam2=args.sam2,
                                device=args.device)
        summary = run(work / "pairs_train", work / "autolabel", labeler, previews=True, verbose=False)
    log["autolabel"] = {k: v for k, v in summary.items() if k != "config"}

    print("3/6 datasets: R1 = ground-truth labels (manual-label upper bound), R2 = PickLabel accepted")
    if args.resume and (work / "datasets" / "R2" / "manifest.json").exists():
        log["R1_dataset"] = read_json(work / "datasets" / "R1" / "manifest.json")
        log["R2_dataset"] = read_json(work / "datasets" / "R2" / "manifest.json")
    else:
        export_gt_run(work / "pairs_train", work / "gt_run")
        log["R1_dataset"] = build([work / "gt_run"], work / "datasets" / "R1", {"accept"}, seed=0)
        log["R2_dataset"] = build([work / "autolabel"], work / "datasets" / "R2", {"accept"}, seed=0)

    print("4/6 train students")
    weights = {}
    for name in ("R1", "R2"):
        best = work / "train" / name / "weights" / "best.pt"
        results_csv = work / "train" / name / "results.csv"
        done = results_csv.exists() and len(results_csv.read_text().splitlines()) - 1 >= args.epochs
        if not (args.resume and best.exists() and done):
            train(work / "datasets" / name / "data.yaml", name, model=args.model, project=work / "train",
                  epochs=args.epochs, imgsz=args.imgsz, batch=8, device=args.device, patience=0)
        weights[name] = best

    print("5/6 evaluate on held-out sessions")
    gts = load_gt(work / "pairs_test")
    curves, points, results = {}, {}, {}
    for name, w in weights.items():
        records = predict_yolo(str(w), [g.image_path for g in gts], args.imgsz, device=args.device)
        rep, curve = evaluate_records(records, gts)
        results[name] = rep.as_dict()
        results[name]["latency_cpu"] = benchmark(str(w), gts[0].image_path, args.imgsz, n=20, warmup=5,
                                                 device="cpu")
        curves[name] = curve
        thr = rep.at_threshold["threshold"]
        points[name] = (thr, f"{name} @ {thr:.2f}")
    log["eval"] = results
    log["human_labelled_instances"] = {
        "R1": log["R1_dataset"]["instances"].get("train", 0) + log["R1_dataset"]["instances"].get("val", 0),
        "R2": 0,
    }
    log["elapsed_min"] = round((time.time() - t0) / 60, 1)

    print("6/6 report")
    report_dir.mkdir(parents=True, exist_ok=True)
    pr_figure({"R1 ground-truth labels": curves["R1"], "R2 PickLabel labels": curves["R2"]},
              report_dir / "pr_synthetic.png", title="Student on held-out synthetic trays (smoke test)",
              points={"R1 ground-truth labels": points["R1"], "R2 PickLabel labels": points["R2"]})
    write_json(report_dir / "synthetic_e2e.json", log)
    print(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
