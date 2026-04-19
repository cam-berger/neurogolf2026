"""LLM-assisted rule synthesis for ARC-AGI tasks.

Calls Claude Opus 4.7 via the Anthropic API to write numpy `method(grid)`
functions from input/output pair examples, tests against all pairs, saves
successful methods to disk for ONNX compilation.

Usage:
    # Key auto-loaded from kaggle-neuro.txt if ANTHROPIC_API_KEY env not set
    python arc_synth.py --smoke-test                 # 2 already-shipped tasks (validates pipeline)
    python arc_synth.py --tasks 20 131 176 190 260   # user's target tasks
    python arc_synth.py --unshipped --limit 20       # 20 unshipped, bulk-test
    python arc_synth.py --unshipped                  # full run (~$10-15)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pipeline.loader import load_task, get_all_pairs


MODEL = "claude-opus-4-7"
MAX_TOKENS = 4096
KEY_FILE = "kaggle-neuro.txt"


SYSTEM_PROMPT = """You are an ARC-AGI task solver. Given input/output pairs where each is a 2D grid of integers 0-9, write a Python function `method(grid)` that transforms any input grid to its matching output grid.

Conventions:
- `grid` is a 2D numpy array of integer color codes (0-9, where 0 is background/black).
- Return a 2D numpy array. Output shape may differ from input.
- Use only numpy (imported as `np`). No external libraries.
- The function must be CORRECT for ALL pairs, not just the 5 shown.
- Think step by step: identify the transformation rule, then implement it.

Common transformation patterns to consider:
- Geometric: rotation, flip, transpose, scaling, cropping
- Color: remapping, swapping, filling based on condition
- Spatial: shifting, gravity, reflection, symmetry completion
- Object-based: keep largest/smallest connected region, count objects
- Pattern: stamp template at markers, extend periodic pattern, flood fill

Output only the Python function in a single ```python``` code block. No prose outside the block."""


USER_TEMPLATE = """Task {task_id} — {n_pairs} training pairs:

{pairs_text}

Write `def method(grid):` that transforms any such input to its output."""


def load_api_key() -> str:
    """Load API key from env or from kaggle-neuro.txt."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    kf = Path(__file__).parent / KEY_FILE
    if kf.is_file():
        return kf.read_text().strip()
    raise RuntimeError(
        f"ANTHROPIC_API_KEY not set and {KEY_FILE} not found. "
        f"Set env var or create {kf.resolve()}."
    )


def format_pair(inp, out, idx: int) -> str:
    def grid_str(g):
        return "\n".join("[" + ", ".join(f"{v:>2d}" for v in row) + "]" for row in g)
    return (f"--- Pair {idx} ---\n"
            f"Input ({len(inp)}x{len(inp[0])}):\n{grid_str(inp)}\n\n"
            f"Output ({len(out)}x{len(out[0])}):\n{grid_str(out)}")


def extract_code(text: str) -> str | None:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def run_method(code: str, task: dict) -> tuple[int, int, int | None, str | None]:
    """Compile and run method. Returns (n_correct, n_total, first_fail_idx, error)."""
    ns: dict = {"np": np}
    try:
        exec(code, ns)
    except Exception as e:
        return 0, 0, None, f"compile error: {e}"

    method = ns.get("method")
    if method is None:
        return 0, 0, None, "no `method` defined"

    pairs = get_all_pairs(task)
    n_correct = 0
    first_fail = None
    for i, (inp, out) in enumerate(pairs):
        try:
            pred = method(np.asarray(inp))
            pred = np.asarray(pred)
            if pred.shape == np.asarray(out).shape and np.array_equal(pred, np.asarray(out)):
                n_correct += 1
            elif first_fail is None:
                first_fail = i
        except Exception:
            if first_fail is None:
                first_fail = i
    return n_correct, len(pairs), first_fail, None


def synthesize(client, task_id: int, max_retries: int = 2, n_sample_pairs: int = 5,
                verbose: bool = True) -> tuple[str | None, str, dict]:
    """Returns (code, status, usage_stats)."""
    task = load_task(task_id)
    pairs = get_all_pairs(task)
    if not pairs:
        return None, "no pairs", {}

    samples = pairs[:n_sample_pairs]
    pairs_text = "\n\n".join(format_pair(i, o, idx) for idx, (i, o) in enumerate(samples))
    user_msg = USER_TEMPLATE.format(task_id=task_id, n_pairs=len(samples), pairs_text=pairs_text)

    messages = [{"role": "user", "content": user_msg}]

    code = None
    last_stats = None
    total_input = total_output = total_cache_read = total_cache_write = 0

    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                thinking={"type": "adaptive"},
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=messages,
            )
        except Exception as e:
            import anthropic
            if isinstance(e, anthropic.RateLimitError):
                wait = 60
                if verbose:
                    print(f"    rate limited — sleeping {wait}s")
                time.sleep(wait)
                continue
            if isinstance(e, anthropic.APIStatusError) and e.status_code >= 500:
                if verbose:
                    print(f"    server error {e.status_code} — retrying in 15s")
                time.sleep(15)
                continue
            return None, f"API error: {e}", {}

        # Track usage
        usage = response.usage
        total_input += usage.input_tokens
        total_output += usage.output_tokens
        total_cache_read += getattr(usage, "cache_read_input_tokens", 0) or 0
        total_cache_write += getattr(usage, "cache_creation_input_tokens", 0) or 0

        response_text = next((b.text for b in response.content if b.type == "text"), "")
        extracted = extract_code(response_text)

        if extracted is None:
            if verbose:
                print(f"    attempt {attempt}: no code block in response")
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user",
                    "content": "Your previous response didn't contain a Python code block. Return only the `method` function in a ```python``` block."})
                continue
            usage_stats = {"input": total_input, "output": total_output,
                           "cache_read": total_cache_read, "cache_write": total_cache_write}
            return None, f"no code after {attempt+1} attempts", usage_stats
        code = extracted

        n_correct, n_total, first_fail, err = run_method(code, task)
        last_stats = (n_correct, n_total, first_fail, err)

        usage_stats = {"input": total_input, "output": total_output,
                       "cache_read": total_cache_read, "cache_write": total_cache_write}

        if err is None and n_correct == n_total:
            return code, f"SUCCESS ({n_total}/{n_total}) attempt={attempt}", usage_stats

        if verbose:
            print(f"    attempt {attempt}: {n_correct}/{n_total} correct"
                  f"{' err=' + err if err else ''}")

        if attempt < max_retries and first_fail is not None:
            fail_inp, fail_out = pairs[first_fail]
            fail_pair_str = format_pair(fail_inp, fail_out, first_fail)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user",
                "content": f"Your function was wrong on pair {first_fail}:\n\n{fail_pair_str}\n\nFix the `method` and return only the corrected function in a ```python``` block."})
        elif attempt < max_retries and err:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user",
                "content": f"Your code errored: {err}. Return only the corrected `method` in a ```python``` block."})

    n_correct, n_total, _, err = last_stats
    return code, f"FAILED ({n_correct}/{n_total}){' err=' + err if err else ''}", usage_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="*", type=int)
    parser.add_argument("--unshipped", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="test on 2 already-shipped tasks (73 ship via CC crop, 150 ship via geometric)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default="llm_synthesis")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Load API key
    api_key = load_api_key()
    os.environ["ANTHROPIC_API_KEY"] = api_key

    import anthropic
    client = anthropic.Anthropic()

    # Determine task list
    if args.smoke_test:
        tasks = [73, 150]   # 73 = local_rule_3x3, 150 = geometric flip_h. Both already shipped.
    elif args.unshipped:
        import glob
        shipped = {int(Path(f).stem[-3:]) for f in glob.glob("output/onnx/task*.onnx")}
        tasks = [t for t in range(1, 401) if t not in shipped]
    elif args.tasks:
        tasks = list(args.tasks)
    else:
        print("Specify --smoke-test, --tasks 20 131, or --unshipped")
        return 1

    if args.limit:
        tasks = tasks[:args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Cost estimate (Opus 4.7: $5/$25 per 1M, cache read ~$0.50/1M, cache write ~$6.25/1M)
    est_cost_low = len(tasks) * 0.02   # best case: cached, one shot
    est_cost_hi = len(tasks) * 0.08    # worst: no cache, 3 attempts each
    print(f"Synthesizing for {len(tasks)} tasks (est ${est_cost_low:.2f}-${est_cost_hi:.2f} API cost)")
    print(f"Model: {MODEL}  Retries: {args.retries}")
    print()

    verbose = not args.quiet
    results = []
    t_start = time.time()
    total_input = total_output = total_cache_read = total_cache_write = 0

    for i, tid in enumerate(tasks):
        t0 = time.time()
        print(f"[{i+1}/{len(tasks)}] Task {tid}...", flush=True)
        try:
            code, status, usage_stats = synthesize(client, tid, max_retries=args.retries,
                                                     verbose=verbose)
        except Exception as e:
            code, status, usage_stats = None, f"EXCEPTION: {e}", {}
        elapsed = time.time() - t0

        total_input += usage_stats.get("input", 0)
        total_output += usage_stats.get("output", 0)
        total_cache_read += usage_stats.get("cache_read", 0)
        total_cache_write += usage_stats.get("cache_write", 0)

        print(f"  -> {status} ({elapsed:.0f}s)", flush=True)

        results.append({"task_id": tid, "status": status, "elapsed": elapsed,
                        "code_saved": code is not None, "usage": usage_stats})
        if code is not None:
            (out_dir / f"task{tid:03d}_method.py").write_text(code + "\n")

    total_time = time.time() - t_start
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # Cost accounting
    cost = (total_input * 5.0 + total_output * 25.0
            + total_cache_read * 0.5 + total_cache_write * 6.25) / 1_000_000

    print(f"\n=== Summary ({total_time:.0f}s, ~${cost:.3f}) ===")
    succ = sum(1 for r in results if r["status"].startswith("SUCCESS"))
    fail = sum(1 for r in results if r["status"].startswith("FAILED"))
    err = len(results) - succ - fail
    print(f"Total: {len(results)}  SUCCESS: {succ}  FAILED: {fail}  ERROR: {err}")
    print(f"Tokens — input: {total_input:,}  output: {total_output:,}  "
          f"cache_read: {total_cache_read:,}  cache_write: {total_cache_write:,}")
    print(f"Saved methods to: {out_dir}/task*_method.py")

    return 0 if succ > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
