"""Enhanced gradient trainer for 5×5 tasks that need more capacity.

The base LocalRuleGenerator uses 10 hidden channels — too few for complex
5×5 rules. This trainer sweeps hidden sizes up to a budget that guarantees
the exported ONNX fits under 1.5 MB, and runs longer (more steps, restarts).
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import onnx

from pipeline.loader import CHANNELS, GRID_SHAPE, encode_grid, get_all_pairs


_OPSET = 10
_FILESIZE_LIMIT = 1_509_949


def _max_hidden(kernel: int) -> int:
    per_filter_bytes = (CHANNELS * kernel * kernel + CHANNELS) * 4 + 100
    return int((_FILESIZE_LIMIT - 10000) / per_filter_bytes)


def _encode_pairs(pairs):
    xs = np.concatenate([encode_grid(inp) for inp, _ in pairs], axis=0)
    ys = np.concatenate([encode_grid(out) for _, out in pairs], axis=0)
    return xs, ys


def train_task(task: dict, kernel: int = 5,
               hidden_sizes: list[int] | None = None,
               n_steps: int = 15000,
               n_restarts: int = 3,
               lr: float = 5e-3,
               margin: float = 0.5,
               verbose: bool = True) -> onnx.ModelProto | None:
    import torch
    import torch.nn as nn

    pool = get_all_pairs(task)
    if not pool:
        return None

    xs_np, ys_np = _encode_pairs(pool)
    xs = torch.from_numpy(xs_np).float()
    ys = torch.from_numpy(ys_np).float()
    sign = 2.0 * ys - 1.0

    max_h = _max_hidden(kernel)
    if hidden_sizes is None:
        hidden_sizes = [h for h in [10, 20, 50, 100, 200, 400] if h <= max_h]

    pad = kernel // 2
    best_model = None
    best_errors = float("inf")
    best_h = None

    for h in hidden_sizes:
        for restart in range(n_restarts):
            net = nn.Sequential(
                nn.Conv2d(CHANNELS, h, kernel_size=kernel, padding=pad, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(h, CHANNELS, kernel_size=1, bias=False),
            )
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

            run_best_errors = len(pool)
            run_best_state = None

            for step in range(n_steps):
                opt.zero_grad()
                logits = net(xs)
                hinge = torch.clamp(margin - sign * logits, min=0.0)
                loss = hinge.mean()
                l1 = sum(p.abs().sum() for p in net.parameters()) * 1e-5
                (loss + l1).backward()
                opt.step()
                sched.step()

                if step % 200 == 0 or step == n_steps - 1:
                    with torch.no_grad():
                        preds = (net(xs) > 0.0).float()
                        errors = int((preds != ys).any(dim=(1, 2, 3)).sum().item())
                        cell_wrong = int((preds != ys).sum().item())
                        total_cells = int(ys.numel())
                    if errors < run_best_errors:
                        run_best_errors = errors
                        run_best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                    if verbose and step % 1000 == 0:
                        print(f"    step {step}: pair_err={errors}/{len(pool)} cell_err={cell_wrong}/{total_cells} ({100*cell_wrong/total_cells:.2f}%)")
                    if errors == 0:
                        break

            if run_best_errors < best_errors and run_best_state is not None:
                best_errors = run_best_errors
                best_model = nn.Sequential(
                    nn.Conv2d(CHANNELS, h, kernel_size=kernel, padding=pad, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(h, CHANNELS, kernel_size=1, bias=False),
                )
                best_model.load_state_dict(run_best_state)
                best_h = h

            if verbose:
                print(f"  h={h} restart={restart}: {run_best_errors} errors (best overall: {best_errors})")

            if best_errors == 0:
                break
        if best_errors == 0:
            break

    if best_errors > 0:
        if verbose:
            print(f"  FAILED: best was {best_errors} errors with h={best_h}")
        return None

    with torch.no_grad():
        final = (best_model(xs).numpy() > 0.0).astype(np.float32)
    if not np.array_equal(final, ys_np):
        if verbose:
            print("  final verification failed")
        return None

    dummy = torch.zeros(GRID_SHAPE, dtype=torch.float32)
    with NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.onnx.export(
            best_model, dummy, tmp_path,
            input_names=["input"], output_names=["output"],
            opset_version=_OPSET, do_constant_folding=True,
        )
        proto = onnx.load(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return proto


__all__ = ["train_task"]
