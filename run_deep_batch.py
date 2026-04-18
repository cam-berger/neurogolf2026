"""Train all remaining 5x5 tasks with deep trainer, longer + more restarts."""
import sys; sys.path.insert(0, '.')
import onnx
import time
from custom_rules.deep_trainer import train_deep
from pipeline.loader import load_task
from pipeline.validator import check_correctness, validate_constraints, compute_cost

# Remaining 5x5 tasks (16 - 1 already shipped = 15)
REMAINING = [129, 167, 389, 278, 329, 298, 12, 369, 70, 359, 243, 265, 162, 77, 208]

shipped = []
failed = []

for tid in REMAINING:
    t0 = time.time()
    print(f"\n=== Task {tid} ===", flush=True)
    task = load_task(tid)
    # Progressive: start with small hidden, fast; escalate if close
    configs = [
        (32, 4000, 2),  # hidden=32, 4000 steps, 2 restarts
        (64, 5000, 2),  # bump up if close
    ]
    best_model = None
    for hidden, n_steps, n_restarts in configs:
        model = train_deep(task, kernel=5, hidden=hidden, n_steps=n_steps,
                            n_restarts=n_restarts, verbose=True)
        if model is not None:
            best_model = model
            break
        print(f"  hidden={hidden}: failed, trying next", flush=True)

    elapsed = time.time() - t0
    if best_model is None:
        print(f"  FAILED ({elapsed:.0f}s)", flush=True)
        failed.append(tid)
        continue

    path = f"/tmp/_deep_{tid:03d}.onnx"
    onnx.save(best_model, path)
    v = validate_constraints(path)
    c = check_correctness(path, task)
    cost = compute_cost(path) if v["valid"] else {"valid": False}
    score = cost.get("score", "n/a") if cost.get("valid") else "n/a"
    print(f"  Result: {c['n_correct']}/{c['n_pairs']}  valid={v['valid']}  {v.get('file_size',0)/1024:.1f}KB  score={score}  ({elapsed:.0f}s)", flush=True)

    if v["valid"] and c["correct"]:
        shipped.append(tid)
        onnx.save(best_model, f"output/onnx/task{tid:03d}.onnx")
        with open(f"custom_rules/task{tid:03d}.py", "w") as f:
            f.write(f'"""Task {tid} — deep trainer 5x5."""\n\n'
                    f'from __future__ import annotations\n\nimport onnx\n\n'
                    f'from custom_rules.deep_trainer import train_deep\n\n\n'
                    f'def generate(task: dict, features: dict) -> onnx.ModelProto | None:\n'
                    f'    return train_deep(task, kernel=5, hidden=32, n_steps=4000, n_restarts=2, verbose=False)\n')
        print(f"  SHIPPED task{tid:03d}", flush=True)

print(f"\n===")
print(f"SHIPPED: {shipped} ({len(shipped)})")
print(f"FAILED: {failed} ({len(failed)})")
