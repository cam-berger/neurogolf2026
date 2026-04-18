"""Warmstart trainer batch on remaining 5x5 tasks."""
import sys; sys.path.insert(0, '.')
import onnx
import time
import os
from custom_rules.warmstart_trainer import train_warmstart
from pipeline.loader import load_task
from pipeline.validator import check_correctness, validate_constraints, compute_cost

# Remaining failing 5x5 tasks
REMAINING = [12, 70, 77, 129, 162, 167, 208, 243, 265, 278, 298, 329, 359, 369]
REMAINING = [t for t in REMAINING if not os.path.exists(f"output/onnx/task{t:03d}.onnx")]

print(f"Warmstart training on: {REMAINING}", flush=True)

shipped = []
for tid in REMAINING:
    t0 = time.time()
    task = load_task(tid)
    print(f"\n=== Task {tid} ===", flush=True)
    # Try progressively larger student sizes
    for h_target in [100, 200, 500]:
        model = train_warmstart(task, kernel=5, hidden_target=h_target, n_steps=4000, lr=2e-3, verbose=True)
        if model is not None:
            break
    elapsed = time.time() - t0
    if model is None:
        print(f"  FAILED ({elapsed:.0f}s)", flush=True)
        continue
    path = f"/tmp/_ws_{tid:03d}.onnx"
    onnx.save(model, path)
    v = validate_constraints(path)
    c = check_correctness(path, task)
    cost = compute_cost(path) if v["valid"] else {"valid": False}
    score = cost.get("score", "n/a") if cost.get("valid") else "n/a"
    print(f"  {c['n_correct']}/{c['n_pairs']}  valid={v['valid']}  {v.get('file_size',0)/1024:.1f}KB  score={score}  ({elapsed:.0f}s)", flush=True)
    if v["valid"] and c["correct"]:
        shipped.append(tid)
        onnx.save(model, f"output/onnx/task{tid:03d}.onnx")
        with open(f"custom_rules/task{tid:03d}.py", "w") as f:
            f.write(f'"""Task {tid} — warmstart trainer."""\n\n'
                    f'from __future__ import annotations\nimport onnx\n'
                    f'from custom_rules.warmstart_trainer import train_warmstart\n\n\n'
                    f'def generate(task: dict, features: dict) -> onnx.ModelProto | None:\n'
                    f'    return train_warmstart(task, kernel=5, hidden_target={h_target}, n_steps=4000, lr=2e-3, verbose=False)\n')
        print(f"  SHIPPED task{tid:03d}", flush=True)

print(f"\nSHIPPED: {shipped} ({len(shipped)})")
