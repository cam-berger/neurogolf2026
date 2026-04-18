"""Deeper Conv network trainer for 5x5 tasks that plateau on 2-layer."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import onnx

from pipeline.loader import CHANNELS, GRID_SHAPE, encode_grid, get_all_pairs


_OPSET = 10
_FILESIZE_LIMIT = 1_509_949


def train_deep(task: dict, kernel: int = 5,
               hidden: int = 32,
               n_steps: int = 5000,
               n_restarts: int = 3,
               lr: float = 3e-3,
               margin: float = 1.0,
               verbose: bool = True,
               batch_size: int = 32) -> onnx.ModelProto | None:
    """3-layer Conv network: Conv(kxk) → ReLU → Conv(3x3) → ReLU → Conv(1x1)."""
    import torch
    import torch.nn as nn

    pool = get_all_pairs(task)
    if not pool:
        return None

    xs_np = np.concatenate([encode_grid(inp) for inp, _ in pool], axis=0)
    ys_np = np.concatenate([encode_grid(out) for _, out in pool], axis=0)
    xs_all = torch.from_numpy(xs_np).float()
    ys_all = torch.from_numpy(ys_np).float()
    n_pairs = xs_all.shape[0]

    pad1 = kernel // 2
    pad2 = 1  # 3x3 middle layer
    best_model = None
    best_errors = float("inf")

    for restart in range(n_restarts):
        net = nn.Sequential(
            nn.Conv2d(CHANNELS, hidden, kernel_size=kernel, padding=pad1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=pad2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, CHANNELS, kernel_size=1, bias=False),
        )
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

        run_best_errors = n_pairs
        run_best_state = None

        for step in range(n_steps):
            # Mini-batch sampling for speed
            idx = torch.randint(0, n_pairs, (batch_size,))
            xs = xs_all[idx]
            ys = ys_all[idx]
            sign = 2.0 * ys - 1.0

            opt.zero_grad()
            logits = net(xs)
            hinge = torch.clamp(margin - sign * logits, min=0.0)
            loss = hinge.mean()
            l1 = sum(p.abs().sum() for p in net.parameters()) * 1e-6
            (loss + l1).backward()
            opt.step()
            sched.step()

            if step % 200 == 0 or step == n_steps - 1:
                with torch.no_grad():
                    preds = (net(xs_all) > 0.0).float()
                    errors = int((preds != ys_all).any(dim=(1, 2, 3)).sum().item())
                    cell_wrong = int((preds != ys_all).sum().item())
                if errors < run_best_errors:
                    run_best_errors = errors
                    run_best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                if verbose and step % 500 == 0:
                    print(f"    r{restart} s{step}: pair_err={errors}/{n_pairs} cell_err={cell_wrong}", flush=True)
                if errors == 0:
                    break

        if run_best_errors < best_errors and run_best_state is not None:
            best_errors = run_best_errors
            best_model = nn.Sequential(
                nn.Conv2d(CHANNELS, hidden, kernel_size=kernel, padding=pad1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=pad2, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, CHANNELS, kernel_size=1, bias=False),
            )
            best_model.load_state_dict(run_best_state)

        if best_errors == 0:
            break

    if best_errors > 0:
        if verbose:
            print(f"  FAILED: {best_errors} pair errors remain")
        return None

    with torch.no_grad():
        final = (best_model(xs_all).numpy() > 0.0).astype(np.float32)
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


__all__ = ["train_deep"]
