"""Collect evaluation reports into the results table and the headline chart.

``runs.yaml`` lists what goes in the table, so the README numbers always trace back to a
report file::

    - name: SAM 3, default thr 0.75
      sam3_report: report/sam3/report.json     # row: best prompt set, node mode, at 0.75
      at_default: true
      labeling_minutes: 0
    - name: R2 PickLabel accept
      report: report/eval/R2/report.json       # from eval-student --latency
      labeling_minutes: 0
    - name: R1 manual
      report: report/eval/R1/report.json
      labeling_minutes: 185                    # stopwatch, not an estimate

    picklabel report runs.yaml --out report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from picklabel.eval.plots import headline_figure
from picklabel.io import read_json


def _row(entry: dict) -> dict:
    if "sam3_report" in entry:
        rows = read_json(entry["sam3_report"])["rows"]
        best = max(rows, key=lambda r: r["recall_at_p90"])
        r = next(x for x in rows if x["prompt_set"] == best["prompt_set"] and x["mode"] == ("node" if entry.get(
            "at_default") else best["mode"]))
        op = r["at_0.75"] if entry.get("at_default") else r["at_best"]
        return {"recall": op["recall"], "precision": op["precision"], "thr": op["threshold"], "ap50": r["ap50"],
                "ms": r["ms_per_frame_p50"]}
    rep = read_json(entry["report"])
    lat = rep.get("latency", {})
    return {"recall": rep["recall_at_p90"], "precision": rep["at_threshold"]["precision"],
            "thr": rep["thr_at_p90"], "ap50": rep["ap50"], "ms": lat.get("ms_p50")}


def build_table(entries: list[dict]) -> tuple[str, list[tuple[str, float, float]]]:
    lines = ["| Model | Recall @ P>=0.90 (or at its threshold) | Precision | Threshold | AP50 (mask) | ms / frame "
             "| Labelling minutes |", "|---|---|---|---|---|---|---|"]
    points = []
    for e in entries:
        r = _row(e)
        fmt = lambda v, d=3: "–" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))  # noqa: E731
        lines.append(f"| {e['name']} | {fmt(r['recall'])} | {fmt(r['precision'])} | {fmt(r['thr'], 2)} | "
                     f"{fmt(r['ap50'])} | {fmt(r['ms'], 1)} | {fmt(e.get('labeling_minutes'), 0)} |")
        if e.get("labeling_minutes") is not None:
            points.append((e["name"], float(e["labeling_minutes"]), float(r["recall"])))
    return "\n".join(lines), points


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Results table + headline chart from evaluation reports.")
    ap.add_argument("runs", help="runs.yaml")
    ap.add_argument("--out", default="report")
    args = ap.parse_args(argv)
    with open(args.runs) as f:
        entries = yaml.safe_load(f)
    table, points = build_table(entries)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results_table.md").write_text(table + "\n")
    if points:
        headline_figure(points, out / "figures" / "headline.png")
    print(table)


if __name__ == "__main__":
    main()
