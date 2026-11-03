"""``picklabel <command> [args]``: one entry point for every stage of the loop."""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    # data
    "synth": ("picklabel.data.synth", "main", "generate synthetic pick sessions with exact ground truth"),
    "capture": ("picklabel.data.capture", "main", "record before/after pairs + grasp points from a camera"),
    "import-photos": ("picklabel.data.capture", "main_import", "alternating before/after photos -> pick folders"),
    "annotate": ("picklabel.data.capture", "main_annotate", "click grasp points for imported pairs"),
    "download": ("picklabel.data.download", "main", "download WS2 / ZeroWaste-f"),
    # data engine
    "autolabel": ("picklabel.autolabel.run", "main", "auto-label pairs (and push to Roboflow)"),
    "eval-autolabel": ("picklabel.eval.autolabel_eval", "main", "gold IoU of auto-labels + routing cutoffs"),
    # model loop
    "build-dataset": ("picklabel.train.build_dataset", "main", "auto-label runs -> YOLO-seg dataset"),
    "copy-paste": ("picklabel.train.copy_paste", "main", "copy-paste augmentation into a dataset"),
    "train": ("picklabel.train.train", "main", "train a YOLO-seg student"),
    "sam3-baseline": ("picklabel.eval.sam3_baseline", "main", "SAM 3 prompt-set and threshold sweep"),
    "eval-student": ("picklabel.eval.eval_student", "main", "evaluate (and time) a student"),
    "gallery": ("picklabel.eval.failure_gallery", "main", "FP/FN crops for failure tagging"),
    "thresholds": ("picklabel.eval.thresholds", "main", "per-class / per-size thresholds"),
    "tiled": ("picklabel.eval.tiled_infer", "main", "full-frame vs tiled inference on small items"),
    "select-uncertain": ("picklabel.eval.active", "main", "active learning: most uncertain frames -> review"),
    "report": ("picklabel.eval.report", "main", "results table + headline chart"),
    "export": ("picklabel.deploy.export", "main", "ONNX export (seg, or ZED custom YOLO)"),
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        width = max(len(c) for c in COMMANDS)
        print("usage: picklabel <command> [args]\n\ncommands:")
        for name, (_, _, help_text) in COMMANDS.items():
            print(f"  {name:<{width}}  {help_text}")
        return
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        raise SystemExit(f"unknown command {name!r}; run `picklabel --help`")
    module, func, _ = COMMANDS[name]
    getattr(importlib.import_module(module), func)(rest)


if __name__ == "__main__":
    main()
