"""Train a YOLO-seg student with Ultralytics on a PickLabel dataset or a Roboflow version.

The run presets in ``configs/train.yaml`` mirror the planned experiments:

* R1: manual labels only (the upper reference)
* R2: PickLabel accepted labels only
* R3: PickLabel accepted + reviewed
* R4: R3 + ZeroWaste-f plastic (does real MRF data help on the tray?)
* R5: R3 with n / s / m model sizes (speed vs accuracy)

    picklabel train --data datasets/R2/data.yaml --preset R2
    picklabel train --roboflow-version 3 --preset R3 --epochs 150

Every run writes ``picklabel_run.json`` next to Ultralytics' outputs: the dataset manifest
and the human labelling minutes that went into it, which the headline chart plots.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from picklabel.io import read_json, write_json

PRESETS_PATH = Path(__file__).resolve().parents[2] / "configs" / "train.yaml"


def load_preset(name: str | None, path: Path = PRESETS_PATH) -> dict:
    if not name:
        return {}
    with open(path) as f:
        presets = yaml.safe_load(f) or {}
    if name not in presets:
        raise SystemExit(f"unknown preset {name!r}; choose from {sorted(presets)}")
    return dict(presets[name])


def download_roboflow(version: int, fmt: str = "yolov11", location: str | None = None) -> Path:
    """Export a Roboflow dataset version and return its data.yaml."""
    from dotenv import load_dotenv
    from roboflow import Roboflow

    load_dotenv()
    project = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"]).workspace(os.environ["ROBOFLOW_WORKSPACE"]).project(
        os.environ["ROBOFLOW_PROJECT"])
    ds = project.version(int(version)).download(fmt, location=location)
    return Path(ds.location) / "data.yaml"


def default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def train(data: str | Path, name: str, model: str = "yolo11s-seg.pt", project: str | Path = "runs/train",
          epochs: int = 100, imgsz: int = 1024, batch: int = 8, device: str | None = None, seed: int = 0,
          patience: int = 20, labeling_minutes: float | None = None, **overrides) -> Path:
    from ultralytics import YOLO

    m = YOLO(model)
    m.train(data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, device=device or default_device(),
            project=str(Path(project).resolve()), name=name, seed=seed, patience=patience, cos_lr=True,
            close_mosaic=min(10, max(epochs // 5, 1)), exist_ok=True, plots=True, verbose=False, **overrides)
    run_dir = Path(m.trainer.save_dir)
    manifest = Path(data).parent / "manifest.json"
    write_json(run_dir / "picklabel_run.json", {
        "name": name, "data": str(data), "model": model, "epochs": epochs, "imgsz": imgsz, "batch": batch,
        "seed": seed, "labeling_minutes": labeling_minutes,
        "dataset_manifest": read_json(manifest) if manifest.exists() else None,
        "best": str(run_dir / "weights" / "best.pt"),
    })
    return run_dir


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train a YOLO-seg student.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="YOLO data.yaml (e.g. from picklabel build-dataset)")
    src.add_argument("--roboflow-version", type=int, help="download this Roboflow dataset version")
    ap.add_argument("--preset", default=None, help="R1..R5 from configs/train.yaml")
    ap.add_argument("--name", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--project", default="runs/train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labeling-minutes", type=float, default=None,
                    help="stopwatch minutes of human labelling/review behind this dataset")
    args = ap.parse_args(argv)

    cfg = load_preset(args.preset)
    data = Path(args.data) if args.data else download_roboflow(args.roboflow_version)
    for key in ("model", "epochs", "imgsz", "batch"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    cfg.pop("description", None)
    name = args.name or args.preset or "student"
    run_dir = train(data, name, project=args.project, device=args.device, seed=args.seed,
                    labeling_minutes=args.labeling_minutes, **cfg)
    print(json.dumps({"run": str(run_dir), "best": str(run_dir / "weights" / "best.pt")}, indent=2))


if __name__ == "__main__":
    main()
