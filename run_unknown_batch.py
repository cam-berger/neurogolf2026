"""Train deep nets on small same-shape unknown tasks."""
import sys; sys.path.insert(0, '.')
import onnx
import time
from custom_rules.deep_trainer import train_deep
from pipeline.loader import load_task
from pipeline.validator import check_correctness, validate_constraints, compute_cost

TARGETS = [
    # Small same-shape, 2-3 colors - easiest for trainer
    10, 27, 42, 43, 156, 169, 176, 226, 248, 254, 273, 277, 292, 299, 320,
    330, 332, 336, 348, 181,
]

shipped = []
failed = []

for tid in TARGETS:
    t0 = time.time()
    print(f"\n=== Task {tid} ===", flush=True)
    task = load_task(tid)
    # Start small, escalate if close
    model = train_deep(task, kernel=5, hidden=16, n_steps=2000, n_restarts=2, verbose=False)
    if model is None:
        # Try larger
        print(f"  small failed, trying hidden=48", flush=True)
        model = train_deep(task, kernel=5, hidden=48, n_steps=3000, n_restarts=1, verbose=False)
    elapsed = time.time() - t0
    if model is None:
        print(f"  FAILED ({elapsed:.0f}s)", flush=True)
        failed.append(tid)
        continue
    path = f"/tmp/_unk_{tid:03d}.onnx"
    onnx.save(model, path)
    v = validate_constraints(path)
    c = check_correctness(path, task)
    cost = compute_cost(path) if v["valid"] else {"valid": False}
    score = cost.get("score", "n/a") if cost.get("valid") else "n/a"
    print(f"  {c['n_correct']}/{c['n_pairs']}  valid={v['valid']}  score={score}  ({elapsed:.0f}s)", flush=True)
    if v["valid"] and c["correct"]:
        shipped.append(tid)
        onnx.save(model, f"output/onnx/task{tid:03d}.onnx")
        with open(f"custom_rules/task{tid:03d}.py", "w") as f:
            f.write(f'"""Task {tid} — deep trainer (unknown)."""\n\n'
                    f'from __future__ import annotations\n\nimport onnx\n\n'
                    f'from custom_rules.deep_trainer import train_deep\n\n\n'
                    f'def generate(task: dict, features: dict) -> onnx.ModelProto | None:\n'
                    f'    return train_deep(task, kernel=5, hidden=16, n_steps=2000, n_restarts=2, verbose=False)\n')
        print(f"  SHIPPED task{tid:03d}", flush=True)

print(f"\n===")
print(f"SHIPPED: {shipped} ({len(shipped)})")
print(f"FAILED: {failed} ({len(failed)})")
