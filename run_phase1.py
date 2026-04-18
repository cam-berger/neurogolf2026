"""Phase 1 CLI: classify every available task and emit a summary table + JSON."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

from classifier.families import TransformFamily
from classifier.features import extract_features
from classifier.rules import classify_task
from pipeline.loader import PROJECT_ROOT, load_task


_TASK_RE = re.compile(r"task(\d{3})\.json$")


def discover_task_ids(root: Path = PROJECT_ROOT) -> list[int]:
    ids = []
    for p in sorted(root.glob("task*.json")):
        m = _TASK_RE.search(p.name)
        if m:
            ids.append(int(m.group(1)))
    return ids


def _feature_digest(features: dict) -> str:
    flags = []
    if features.get("output_shape_eq_input"):
        flags.append("same-shape")
    if features.get("output_is_input_scaled"):
        sh, sw = int(features["scale_factor_h"]), int(features["scale_factor_w"])
        if (sh, sw) != (1, 1):
            flags.append(f"scale{sh}x{sw}")
    if features.get("is_color_permutation"):
        flags.append("color-perm")
    k = features.get("max_local_context_needed", 99)
    if k <= 5:
        flags.append(f"local-k{k}")
    return ", ".join(flags) if flags else "-"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1: classify ARC tasks.")
    parser.add_argument("--output", default="phase1_results.json", help="JSON output path")
    parser.add_argument("--tasks", nargs="*", type=int, help="Optional subset of task ids")
    args = parser.parse_args(argv)

    console = Console()
    task_ids = args.tasks or discover_task_ids()
    console.print(f"[bold]Classifying {len(task_ids)} tasks...[/bold]")

    rows: list[dict] = []
    family_counts: Counter = Counter()

    for tid in task_ids:
        try:
            task = load_task(tid)
        except FileNotFoundError:
            continue
        features = extract_features(task)
        family = classify_task(features)
        family_counts[family] += 1
        rows.append({
            "task_id": tid,
            "family": family.value,
            "features": {
                "output_shape_eq_input": features.get("output_shape_eq_input"),
                "is_color_permutation": features.get("is_color_permutation"),
                "color_permutation_map": features.get("color_permutation_map"),
                "output_is_input_scaled": features.get("output_is_input_scaled"),
                "scale_factor_h": features.get("scale_factor_h"),
                "scale_factor_w": features.get("scale_factor_w"),
                "max_local_context_needed": features.get("max_local_context_needed"),
                "input_max_h": features.get("input_max_h"),
                "input_max_w": features.get("input_max_w"),
                "output_max_h": features.get("output_max_h"),
                "output_max_w": features.get("output_max_w"),
            },
            "digest": _feature_digest(features),
        })

    table = Table(title="Phase 1 Classification")
    table.add_column("Task")
    table.add_column("Family")
    table.add_column("Features")
    for row in rows:
        table.add_row(str(row["task_id"]), row["family"], row["digest"])
    console.print(table)

    console.print("\n[bold]Family breakdown[/bold]")
    for family, count in family_counts.most_common():
        console.print(f"  {family.value:>18s}  {count}")

    out_path = PROJECT_ROOT / args.output
    out_path.write_text(json.dumps({"rows": rows, "counts": {f.value: family_counts[f] for f in family_counts}}, indent=2))
    console.print(f"\n[green]Saved -> {out_path}[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
