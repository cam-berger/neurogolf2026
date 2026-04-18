"""Fast filter for unknown tasks: quick 500-step training to see which are learnable."""
import sys; sys.path.insert(0, '.')
import json
import onnx
import time
import numpy as np
from custom_rules.deep_trainer import train_deep
from pipeline.loader import load_task, get_all_pairs
from pipeline.validator import check_correctness, validate_constraints, compute_cost

p1 = json.load(open('phase1_results.json'))
unclassified = sorted(r['task_id'] for r in p1['rows'] if r.get('family') == 'unknown')

# Same-shape, small-ish
candidates = []
for tid in unclassified:
    task = load_task(tid)
    pairs = get_all_pairs(task)
    if not pairs: continue
    if not all(len(i) == len(o) and len(i[0]) == len(o[0]) for i, o in pairs):
        continue
    max_size = max(len(i) * len(i[0]) for i, _ in pairs)
    if max_size <= 150:
        candidates.append(tid)

print(f"Trying {len(candidates)} same-shape small unknown tasks")

shipped = []
for tid in candidates:
    t0 = time.time()
    task = load_task(tid)
    # Fast quick test first
    model = train_deep(task, kernel=5, hidden=16, n_steps=800, n_restarts=1, verbose=False)
    elapsed = time.time() - t0
    if model is None:
        print(f"task {tid}: no-conv ({elapsed:.0f}s)", flush=True)
        continue
    # Quick test passed, verify
    path = f"/tmp/_u2_{tid:03d}.onnx"
    onnx.save(model, path)
    v = validate_constraints(path)
    c = check_correctness(path, task)
    if v["valid"] and c["correct"]:
        cost = compute_cost(path)
        print(f"task {tid}: SHIP {c['n_correct']}/{c['n_pairs']} score={cost['score']:.2f} ({elapsed:.0f}s)", flush=True)
        shipped.append(tid)
        onnx.save(model, f"output/onnx/task{tid:03d}.onnx")
        with open(f"custom_rules/task{tid:03d}.py", "w") as f:
            f.write(f'"""Task {tid} — deep trainer (unknown)."""\n\n'
                    f'from __future__ import annotations\n\nimport onnx\n\n'
                    f'from custom_rules.deep_trainer import train_deep\n\n\n'
                    f'def generate(task: dict, features: dict) -> onnx.ModelProto | None:\n'
                    f'    return train_deep(task, kernel=5, hidden=16, n_steps=800, n_restarts=1, verbose=False)\n')
    else:
        print(f"task {tid}: train ok but wrong {c['n_correct']}/{c['n_pairs']} ({elapsed:.0f}s)", flush=True)

print(f"\nSHIPPED: {shipped} ({len(shipped)})")
