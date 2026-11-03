"""Export the student for the robot.

* ``seg``: segmentation ONNX for the PickLabel ROS 2 node, or for TensorRT on the robot's GPU.
* ``zed``: a box-only "detection twin" (a YOLO detector trained on the same Roboflow version)
  exported in the format the ZED SDK's ``CUSTOM_YOLOLIKE_BOX_OBJECTS`` mode reads, plus
  the ``custom_object_detection.yaml`` block for Grip's zed-ros2-wrapper fork. That path
  gives boxes with the ZED's own 3D tracking, but no masks.

    picklabel export --weights runs/train/R3/weights/best.pt --kind seg --imgsz 1024
    picklabel export --weights runs/train/R3_det/weights/best.pt --kind zed --imgsz 512
"""

from __future__ import annotations

import argparse
from pathlib import Path


def export_onnx(weights: str, imgsz: int = 640, dynamic: bool = False, half: bool = False) -> Path:
    from ultralytics import YOLO

    path = YOLO(weights).export(format="onnx", imgsz=imgsz, simplify=True, dynamic=dynamic, half=half)
    return Path(path)


def zed_yaml(class_names: list[str], onnx_path: str, input_size: int, confidence: float = 50.0) -> str:
    """The block for zed_wrapper/config/custom_object_detection.yaml (plus the switch in common_stereo.yaml)."""
    lines = [
        "# common_stereo.yaml:",
        "#   object_detection.od_enabled: true",
        "#   object_detection.detection_model: 'CUSTOM_YOLOLIKE_BOX_OBJECTS'",
        "/**:",
        "  ros__parameters:",
        "    object_detection:",
        f"      custom_onnx_file: '{onnx_path}'",
        f"      custom_onnx_input_size: {input_size}",
        f"      custom_class_count: {len(class_names)}",
    ]
    for i, name in enumerate(class_names):
        lines += [
            f"      class_{i:03d}:",
            f"        label: '{name}'",
            f"        model_class_id: {i}",
            "        enabled: true",
            f"        confidence_threshold: {confidence:.1f}  # [0, 99]",
            "        is_grounded: true   # lies on the belt",
            "        is_static: false    # moves with the belt",
            "        tracking_timeout: -1.0",
            "        tracking_max_dist: -1.0",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Export the student to ONNX (seg) or the ZED custom-YOLO format.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--kind", choices=["seg", "zed"], default="seg")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--confidence", type=float, default=50.0, help="ZED per-class confidence threshold [0, 99]")
    args = ap.parse_args(argv)
    imgsz = args.imgsz or (512 if args.kind == "zed" else 1024)
    if args.kind == "zed":
        from ultralytics import YOLO

        task = YOLO(args.weights).task
        if task != "detect":
            print(f"warning: {args.weights} is a '{task}' model; the ZED SDK expects a box detector. "
                  "Train a detection twin (yolo11s.pt) on the same data (see export/zed_custom_yolo.md).")
    onnx = export_onnx(args.weights, imgsz=imgsz, half=args.half)
    print(f"exported {onnx}")
    if args.kind == "zed":
        names = list(YOLO(args.weights).names.values())
        snippet = zed_yaml(names, str(onnx.resolve()), imgsz, args.confidence)
        out = onnx.with_suffix(".zed_custom_object_detection.yaml")
        out.write_text(snippet)
        print(f"ZED config block -> {out}\n{snippet}")


if __name__ == "__main__":
    main()
