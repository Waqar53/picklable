"""Route-aware uploads to Roboflow.

* ``accept``: annotations uploaded as ground truth; the image joins the dataset.
* ``review``: annotations uploaded as *predictions* and kept out of the dataset
  (``add_to_dataset=False``) until someone approves them in Roboflow.
* ``hard``: the image goes up with any partial labels as predictions, tagged so a human
  labels it first. Failed picks land here: they are where perception was wrong.

Every image carries tags (route, source, quality bucket, session, reasons) and custom
metadata (score and signals), so Roboflow filters and dataset versions can use them. A local
ledger makes re-runs idempotent, and ``dry_run`` records the exact SDK calls instead of
sending them.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from picklabel.types import Route


@dataclass
class UploadItem:
    key: str  # unique per image, e.g. "<session>__<pick_id>"
    image_path: Path
    route: Route
    coco_path: Path | None = None
    n_instances: int = 0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    split: str = "train"


def make_tags(route: Route, source: str, score: float, session: str, reasons: list[str],
              extra: list[str] | None = None) -> list[str]:
    tags = ["picklabel", f"route:{route.value}", f"source:{source}", f"q:{int(score * 10) / 10:.1f}",
            f"session:{session}"]
    tags += [f"reason:{r}" for r in reasons[:4]]
    if route == Route.HARD:
        tags.append("priority:high")
    return tags + list(extra or [])


def default_batch_name(origin: str = "home") -> str:
    return f"picklabel-{time.strftime('%Y%m%d')}-{origin}"


class RoboflowSync:
    def __init__(self, api_key: str | None = None, workspace: str | None = None, project: str | None = None,
                 dry_run: bool = False, ledger_path: str | Path = "roboflow_ledger.jsonl", num_retry: int = 3):
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        self.dry_run = dry_run
        self.num_retry = num_retry
        self.ledger_path = Path(ledger_path)
        self.done: set[str] = set()
        if self.ledger_path.exists():
            for line in self.ledger_path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not rec.get("dry_run") and not rec.get("error"):
                    self.done.add(rec["key"])
        self.project = None
        if not dry_run:
            api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
            workspace = workspace or os.environ.get("ROBOFLOW_WORKSPACE")
            project = project or os.environ.get("ROBOFLOW_PROJECT")
            if not (api_key and workspace and project):
                raise ValueError("Set ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE and ROBOFLOW_PROJECT (see .env.example), "
                                 "or use dry_run=True.")
            import roboflow

            self.project = roboflow.Roboflow(api_key=api_key).workspace(workspace).project(project)

    def call_args(self, item: UploadItem, batch_name: str) -> dict[str, Any]:
        """The exact keyword arguments for ``Project.single_upload``."""
        kwargs: dict[str, Any] = {
            "image_path": str(item.image_path),
            "split": item.split,
            "batch_name": batch_name,
            "tag_names": list(item.tags),
            "metadata": dict(item.metadata),
            "num_retry_uploads": self.num_retry,
        }
        if item.coco_path is not None and item.n_instances > 0:
            kwargs["annotation_path"] = str(item.coco_path)
            if item.route != Route.ACCEPT:
                kwargs["is_prediction"] = True
                kwargs["add_to_dataset"] = False
        return kwargs

    def push(self, item: UploadItem, batch_name: str) -> dict[str, Any]:
        if item.key in self.done:
            return {"key": item.key, "skipped": "already uploaded"}
        kwargs = self.call_args(item, batch_name)
        record: dict[str, Any] = {"key": item.key, "route": item.route.value, "time": time.time()}
        if self.dry_run:
            record.update(dry_run=True, kwargs=kwargs)
        else:
            try:
                result = self.project.single_upload(**kwargs)
                image = result.get("image") or {}
                record["image_id"] = image.get("id") if isinstance(image, dict) else None
                self.done.add(item.key)
            except Exception as e:  # keep going; the ledger shows what failed
                record["error"] = f"{type(e).__name__}: {e}"
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record
