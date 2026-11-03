"""Core data types shared across PickLabel."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class Route(str, Enum):
    """Where an auto-label goes in Roboflow."""

    ACCEPT = "accept"  # confident: uploaded as ground truth
    REVIEW = "review"  # plausible: uploaded as a prediction into the review queue
    HARD = "hard"  # failed or ambiguous pick: prioritised for human labelling

    @property
    def rank(self) -> int:
        return {"accept": 0, "review": 1, "hard": 2}[self.value]

    @staticmethod
    def worst(routes: list[Route]) -> Route:
        return max(routes, key=lambda r: r.rank) if routes else Route.ACCEPT


@dataclass
class PickPair:
    """One pick event: the scene before and after a single item was removed.

    Coordinates are pixels in the *before* frame. ``belt_shift_px`` is the displacement
    of belt content between the two frames (a point at ``p`` before is at ``p + shift``
    after), which a tracking robot already knows from the conveyor encoder.
    """

    pick_id: str
    before: np.ndarray  # BGR uint8, HxWx3
    after: np.ndarray  # BGR uint8, HxWx3
    grasp_xy: tuple[float, float]
    session: str = "default"
    class_name: str = "plastic"
    grasp_success: bool | None = None  # gripper feedback, when the robot reports it
    belt_shift_px: tuple[float, float] | None = None
    timestamp: float | None = None
    gt_mask: np.ndarray | None = None  # gold mask of the picked item (gold set / synthetic)
    gt_instances: np.ndarray | None = None  # uint16 instance-id map of every target in `before`
    meta: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.before.shape[:2]


@dataclass
class AutoLabel:
    """The auto-labeler's verdict for one pick."""

    pick_id: str
    session: str
    class_name: str
    route: Route
    score: float
    mask: np.ndarray | None  # bool HxW in the before frame
    source: str  # sam3 | sam2 | change | none
    reasons: list[str] = field(default_factory=list)
    signals: dict[str, float] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    align: dict[str, Any] = field(default_factory=dict)
    grasp_xy: tuple[float, float] | None = None
    # Debug products, kept only when asked for (previews, the demo app).
    change_score: np.ndarray | None = None
    change_mask: np.ndarray | None = None
    region: np.ndarray | None = None
    after_aligned: np.ndarray | None = None

    def to_record(self) -> dict[str, Any]:
        """A JSON-serialisable summary for manifests and Roboflow metadata."""
        rec: dict[str, Any] = {
            "pick_id": self.pick_id,
            "session": self.session,
            "class_name": self.class_name,
            "route": self.route.value,
            "score": round(float(self.score), 4),
            "source": self.source,
            "reasons": list(self.reasons),
            "signals": {k: round(float(v), 4) for k, v in self.signals.items()},
            "timings_ms": {k: round(float(v), 1) for k, v in self.timings_ms.items()},
            "align": self.align,
            "grasp_xy": [round(float(v), 1) for v in self.grasp_xy] if self.grasp_xy else None,
            "mask_area": int(self.mask.sum()) if self.mask is not None else 0,
        }
        return rec
