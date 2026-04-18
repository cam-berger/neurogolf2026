"""Batch train the 16 remaining 5x5 tasks (conservative)."""
import sys; sys.path.insert(0, '.')
import onnx
import time
from custom_rules.enhanced_trainer import train_task
from pipeline.loader import load_task
from pipeline.validator import check_correctness, validate_constraints, compute_cost

ORDERED = [129, 167, 52, 389, 278, 329, 298, 139, 12, 369, 70, 359, 243, 265, 162, 77, 208]

shipped = []
failed = []

for tid in ORDERED:
    t0 = time.time()
    print(f"\n=== Task {tid} ===", flush=True)
    task = load_task(tid)
    # Start with smaller hidden sizes, longer training
    model = train_task(
        task, kernel=5,
        hidden_sizes=[20, 50, 100],
        n_steps=20000,
        n_restarts=1,
        margin=1.0,
        verbose=True,
    )
    elapsed = time.time() - t0
    if model is None:
        print(f"  Task {tid}: FAILED ({elapsed:.0f}s)", flush=True)
        failed.append(tid)
        continue
    path = f"/tmp/_batch_{tid:03d}.onnx"
    onnx.save(model, path)
    c = check_correctness(path, task)
    v = validate_constraints(path)
    cost = compute_cost(path) if v["valid"] else {"valid": False}
    score = cost.get("score", "n/a") if cost.get("valid") else "n/a"
    print(f"  {c['n_correct']}/{c['n_pairs']}  valid={v['valid']}  {v.get('file_size',0)/1024:.1f}KB  score={score}  ({elapsed:.0f}s)", flush=True)
    if v["valid"] and c["correct"]:
        shipped.append(tid)
        onnx.save(model, f"output/onnx/task{tid:03d}.onnx")
        with open(f"custom_rules/task{tid:03d}.py", "w") as f:
            f.write(f'"""Task {tid} — trained 5x5 generator."""\n\n'
                    f'from __future__ import annotations\n\nimport onnx\n\n'
                    f'from custom_rules.enhanced_trainer import train_task\n\n\n'
                    f'def generate(task: dict, features: dict) -> onnx.ModelProto | None:\n'
                    f'    return train_task(task, kernel=5, hidden_sizes=[20, 50, 100],\n'
                    f'                      n_steps=20000, n_restarts=1, margin=1.0, verbose=False)\n')
        print(f"  SHIPPED task{tid:03d}", flush=True)

print(f"\n===")
print(f"SHIPPED: {shipped} ({len(shipped)})")
print(f"FAILED: {failed} ({len(failed)})")
