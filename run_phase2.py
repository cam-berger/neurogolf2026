"""Phase 2 CLI: generate + validate ONNX for each classifiable task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from classifier.families import TransformFamily
from classifier.features import extract_features
from classifier.rules import classify_task
from generators import GENERATORS
from pipeline.loader import PROJECT_ROOT, load_task
from pipeline.validator import check_correctness, compute_cost, format_cost, validate_constraints

OUTPUT_DIR = PROJECT_ROOT / "output" / "onnx"


def _status(result: dict) -> str:
    if not result["constraints"]["valid"]:
        return "invalid"
    if not result["correctness"]["correct"]:
        return f"wrong ({result['correctness']['n_correct']}/{result['correctness']['n_pairs']})"
    return "ok"


def run_task(task_id: int, console: Console) -> dict:
    task = load_task(task_id)
    features = extract_features(task)
    features["_task_id"] = task_id
    family = classify_task(features)

    record: dict = {"task_id": task_id, "family": family.value, "generated": False}

    if family == TransformFamily.UNKNOWN:
        record["reason"] = "unclassified"
        return record

    generator = GENERATORS.get(family)
    if generator is None:
        record["reason"] = f"no generator for {family.value}"
        return record

    if not generator.can_generate(task, features):
        record["reason"] = "generator pre-conditions unmet (likely varying shape across pairs)"
        return record

    try:
        model = generator.generate(task, features)
    except Exception as e:  # noqa: BLE001
        record["reason"] = f"generate() raised: {e}"
        return record

    if model is None:
        record["reason"] = "generator returned None"
        return record

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"task{task_id:03d}.onnx"
    generator.save(model, str(out_path))

    constraints = validate_constraints(str(out_path))
    correctness = check_correctness(str(out_path), task)
    cost = compute_cost(str(out_path)) if constraints["valid"] else {"valid": False, "reason": "skipped"}

    record.update({
        "generated": True,
        "path": str(out_path.relative_to(PROJECT_ROOT)),
        "constraints": constraints,
        "correctness": correctness,
        "cost": cost,
    })
    record["status"] = _status(record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2: generate ONNX for classifiable tasks.")
    parser.add_argument("--phase1", default="phase1_results.json", help="Phase 1 JSON (optional, for task list)")
    parser.add_argument("--output", default="phase2_results.json", help="JSON output path")
    parser.add_argument("--tasks", nargs="*", type=int, help="Optional subset of task ids")
    args = parser.parse_args(argv)

    console = Console()

    if args.tasks:
        task_ids = args.tasks
    else:
        phase1 = PROJECT_ROOT / args.phase1
        if phase1.is_file():
            data = json.loads(phase1.read_text())
            task_ids = [row["task_id"] for row in data.get("rows", []) if row["family"] != "unknown"]
        else:
            from run_phase1 import discover_task_ids
            task_ids = discover_task_ids()

    console.print(f"[bold]Generating ONNX for {len(task_ids)} tasks...[/bold]")

    records: list[dict] = []
    for tid in task_ids:
        try:
            rec = run_task(tid, console)
        except FileNotFoundError:
            continue
        records.append(rec)

    table = Table(title="Phase 2 Generation")
    table.add_column("Task"); table.add_column("Family")
    table.add_column("Status"); table.add_column("Cost")
    for rec in records:
        cost_str = format_cost(rec.get("cost", {"valid": False, "reason": "n/a"})) if rec.get("generated") else "-"
        status = rec.get("status", rec.get("reason", "-"))
        table.add_row(str(rec["task_id"]), rec["family"], status, cost_str)
    console.print(table)

    ok = sum(1 for r in records if r.get("status") == "ok")
    console.print(f"\n[bold]Passed:[/bold] {ok} / {len(records)}")

    out_path = PROJECT_ROOT / args.output
    out_path.write_text(json.dumps(records, indent=2, default=str))
    console.print(f"[green]Saved -> {out_path}[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
