"""Fast 60s-per-task scan across ALL unclassified same-shape tasks."""
import sys; sys.path.insert(0, '.')
import json
import signal
import onnx
import time
from custom_rules.deep_trainer import train_deep
from pipeline.loader import load_task, get_all_pairs
from pipeline.validator import check_correctness, validate_constraints, compute_cost

p1 = json.load(open('phase1_results.json'))
unclassified = sorted(r['task_id'] for r in p1['rows'] if r.get('family') == 'unknown')

# All same-shape unknown tasks
same_shape = []
for tid in unclassified:
    task = load_task(tid)
    pairs = get_all_pairs(task)
    if pairs and all(len(i) == len(o) and len(i[0]) == len(o[0]) for i, o in pairs):
        same_shape.append(tid)

# Also retry failing 5x5 with deep trainer (the 15 not yet shipped)
failed_5x5 = [129, 167, 389, 278, 329, 298, 12, 369, 70, 359, 243, 265, 162, 77, 208]

all_targets = failed_5x5 + same_shape
print(f"Total targets: {len(all_targets)} ({len(failed_5x5)} failed_5x5 + {len(same_shape)} unknown)", flush=True)

shipped = []

class Timeout(Exception): pass
def _timeout_handler(signum, frame):
    raise Timeout()

signal.signal(signal.SIGALRM, _timeout_handler)

for tid in all_targets:
    t0 = time.time()
    try:
        task = load_task(tid)
    except FileNotFoundError:
        continue
    # Skip if already shipped
    import os
    if os.path.exists(f"output/onnx/task{tid:03d}.onnx"):
        continue

    # Quick 60s max training
    signal.alarm(60)
    try:
        model = train_deep(task, kernel=5, hidden=24, n_steps=1500, n_restarts=1, verbose=False)
    except Timeout:
        model = None
        print(f"task {tid}: timeout (60s)", flush=True)
        signal.alarm(0)
        continue
    signal.alarm(0)

    elapsed = time.time() - t0
    if model is None:
        continue
    path = f"/tmp/_fs_{tid:03d}.onnx"
    onnx.save(model, path)
    v = validate_constraints(path)
    c = check_correctness(path, task)
    if v["valid"] and c["correct"]:
        cost = compute_cost(path)
        print(f"task {tid}: SHIP {c['n_correct']}/{c['n_pairs']} score={cost['score']:.2f} ({elapsed:.0f}s)", flush=True)
        shipped.append(tid)
        onnx.save(model, f"output/onnx/task{tid:03d}.onnx")
        with open(f"custom_rules/task{tid:03d}.py", "w") as f:
            f.write(f'"""Task {tid} — deep trainer."""\n\nfrom __future__ import annotations\nimport onnx\nfrom custom_rules.deep_trainer import train_deep\n\n\n'
                    f'def generate(task: dict, features: dict) -> onnx.ModelProto | None:\n'
                    f'    return train_deep(task, kernel=5, hidden=24, n_steps=1500, n_restarts=1, verbose=False)\n')

print(f"\nSHIPPED: {shipped} ({len(shipped)})")
