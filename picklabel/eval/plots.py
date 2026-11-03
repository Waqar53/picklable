"""Static report figures (PNG for the README): PR curves, calibration, the headline chart.

One axis per chart, thin 2 px lines, categorical colours in a fixed order, recessive grid,
text in ink colours (never the series colour), and direct labels next to the legend.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from picklabel.eval.metrics import PRCurve  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def _axes(title: str, xlabel: str, ylabel: str, size=(6.4, 4.2)):
    fig, ax = plt.subplots(figsize=size, dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def pr_figure(curves: dict[str, PRCurve], path: str | Path, title: str = "Precision vs recall (mask IoU >= 0.5)",
              points: dict[str, tuple[float, str]] | None = None, min_precision: float = 0.9) -> Path:
    """``points[name] = (threshold, label)`` marks an operating point on that curve."""
    fig, ax = _axes(title, "recall", "precision")
    ax.axhline(min_precision, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.01, min_precision + 0.01, f"P = {min_precision:.2f}", color=INK_2, fontsize=8)
    ends = []
    for i, (name, c) in enumerate(curves.items()):
        color = SERIES[i % len(SERIES)]
        # Operating points often sit on top of each other, so their text goes in the legend.
        legend = f"{name}  ({points[name][1]})" if points and name in points else name
        if c.recall.size:
            r = np.concatenate([[0.0], c.recall])
            p = np.concatenate([[c.precision[0]], c.precision])
            ax.plot(r, p, color=color, linewidth=2, label=legend)
            ends.append((name, float(r[-1]), float(p[-1])))
        if points and name in points:
            pp, rr = c.at(points[name][0])
            ax.plot([rr], [pp], marker="o", markersize=8, color=color, markeredgecolor=SURFACE, markeredgewidth=2)
    last = None  # end-of-curve labels, pushed apart so they never overprint
    for name, x, y in sorted(ends, key=lambda e: -e[2]):
        ty = y if last is None else min(y, last - 0.05)
        ax.annotate(name, (x, y), xytext=(x + 0.015, ty), textcoords="data", fontsize=8, color=INK, va="center",
                    arrowprops={"arrowstyle": "-", "color": GRID, "lw": 0.8} if abs(ty - y) > 1e-6 else None)
        last = ty
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, 1.02)
    if len(curves) > 1 or points:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    return _save(fig, path)


def calibration_figure(scores: Sequence[float], ious: Sequence[float], groups: Sequence[str], path: str | Path,
                       accept: float, review: float) -> Path:
    fig, ax = _axes("Auto-label quality score vs true mask IoU", "quality score", "IoU with gold mask")
    names = list(dict.fromkeys(groups))
    s, q = np.asarray(scores), np.asarray(ious)
    g = np.asarray(groups)
    for i, name in enumerate(names):
        sel = g == name
        ax.scatter(s[sel], q[sel], s=36, color=SERIES[i % len(SERIES)], edgecolors=SURFACE, linewidths=1.5,
                   label=f"{name} (n={int(sel.sum())})", zorder=3)
    for x, label in ((review, "review"), (accept, "accept")):
        ax.axvline(x, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(x + 0.005, 0.02, f"{label} >= {x:.2f}", color=INK_2, fontsize=8, rotation=90, va="bottom")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.03)
    if len(names) > 1:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    return _save(fig, path)


def headline_figure(rows: Sequence[tuple[str, float, float]], path: str | Path,
                    title: str = "Recall @ precision 0.90 vs human labelling time",
                    xlabel: str = "human labelling minutes") -> Path:
    """``rows = [(name, human_cost, recall_at_p90), ...]``; cost is minutes, or labelled instances."""
    fig, ax = _axes(title, xlabel, "recall @ P >= 0.90")
    for i, (name, minutes, recall) in enumerate(rows):
        ax.plot([minutes], [recall], marker="o", markersize=9, color=SERIES[i % len(SERIES)],
                markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none", zorder=3)
        ax.annotate(f"{name}  {recall:.2f}", (minutes, recall), xytext=(8, 2), textcoords="offset points",
                    fontsize=8, color=INK)
    xs = [m for _, m, _ in rows] or [1]
    ax.set_xlim(-0.05 * max(xs) - 1, max(xs) * 1.35 + 1)
    ax.set_ylim(0, 1.02)
    return _save(fig, path)
