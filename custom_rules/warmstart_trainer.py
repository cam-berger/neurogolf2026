"""Warmstart trainer: initialize from known-correct LUT, compress via pruning + fine-tune.

For tasks where the 5x5 LUT is correct but too big, we:
1. Build the LUT network (correct but oversize)
2. Copy weights to a smaller-capacity student network
3. Fine-tune on distilled outputs

This is more principled than cold-starting the trainer.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import onnx

from custom_rules.lut import extract_lut, _one_hot_window
from pipeline.loader import CHANNELS, GRID_SHAPE, encode_grid, get_all_pairs


_OPSET = 10


def train_warmstart(task: dict, kernel: int = 5,
                    hidden_target: int = 200,
                    n_steps: int = 5000,
                    lr: float = 1e-3,
                    verbose: bool = True) -> onnx.ModelProto | None:
    """Train student network warmstarted from teacher LUT."""
    import torch
    import torch.nn as nn

    lut = extract_lut(task, kernel)
    if lut is None:
        return None

    # Teacher: build LUT weights
    h_teacher = len(lut)
    pad = kernel // 2

    windows = list(lut.keys())
    teacher_W1 = np.zeros((h_teacher, CHANNELS, kernel, kernel), dtype=np.float32)
    teacher_B1 = np.zeros((h_teacher,), dtype=np.float32)
    teacher_W2 = np.zeros((CHANNELS, h_teacher, 1, 1), dtype=np.float32)
    for idx, win in enumerate(windows):
        teacher_W1[idx] = _one_hot_window(win, kernel)
        n_valid = sum(1 for v in win if 0 <= v < CHANNELS)
        teacher_B1[idx] = -(n_valid - 0.5) if n_valid > 0 else -0.5
        color = lut[win]
        if 0 <= color < CHANNELS:
            teacher_W2[color, idx, 0, 0] = 2.0

    # Student: smaller capacity
    h_student = min(hidden_target, h_teacher // 2)
    if h_student < 10:
        h_student = 10

    pool = get_all_pairs(task)
    xs_np = np.concatenate([encode_grid(inp) for inp, _ in pool], axis=0)
    ys_np = np.concatenate([encode_grid(out) for _, out in pool], axis=0)
    xs = torch.from_numpy(xs_np).float()
    ys = torch.from_numpy(ys_np).float()
    sign = 2.0 * ys - 1.0

    # Initialize student from top-h_student teacher filters (random subset for diversity)
    rng = np.random.default_rng(0)
    idx = rng.choice(h_teacher, size=h_student, replace=False)
    w1_init = teacher_W1[idx]
    b1_init = teacher_B1[idx]
    # Add small noise so gradient can move
    w1_init = w1_init + rng.normal(0, 0.05, w1_init.shape).astype(np.float32)
    w2_init = teacher_W2[:, idx, :, :]

    net = nn.Sequential(
        nn.Conv2d(CHANNELS, h_student, kernel_size=kernel, padding=pad, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(h_student, CHANNELS, kernel_size=1, bias=False),
    )
    with torch.no_grad():
        net[0].weight.copy_(torch.from_numpy(w1_init))
        net[0].bias.copy_(torch.from_numpy(b1_init))
        net[2].weight.copy_(torch.from_numpy(w2_init))

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    margin = 1.0

    best_errors = float("inf")
    best_state = None
    n_pairs = xs.shape[0]

    for step in range(n_steps):
        opt.zero_grad()
        # Mini-batch
        idx_b = torch.randint(0, n_pairs, (min(64, n_pairs),))
        logits = net(xs[idx_b])
        hinge = torch.clamp(margin - sign[idx_b] * logits, min=0.0)
        loss = hinge.mean()
        loss.backward()
        opt.step()
        sched.step()

        if step % 200 == 0 or step == n_steps - 1:
            with torch.no_grad():
                preds = (net(xs) > 0.0).float()
                errors = int((preds != ys).any(dim=(1, 2, 3)).sum().item())
                cell_wrong = int((preds != ys).sum().item())
            if errors < best_errors:
                best_errors = errors
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            if verbose and step % 500 == 0:
                print(f"  s{step}: pair_err={errors}/{n_pairs} cell_err={cell_wrong} h={h_student}", flush=True)
            if errors == 0:
                break

    if best_state is None or best_errors > 0:
        if verbose:
            print(f"  FAILED: {best_errors} pair errors", flush=True)
        return None

    net.load_state_dict(best_state)
    with torch.no_grad():
        final = (net(xs).numpy() > 0.0).astype(np.float32)
    if not np.array_equal(final, ys_np):
        return None

    dummy = torch.zeros(GRID_SHAPE, dtype=torch.float32)
    with NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.onnx.export(
            net, dummy, tmp_path,
            input_names=["input"], output_names=["output"],
            opset_version=_OPSET, do_constant_folding=True,
        )
        proto = onnx.load(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return proto


__all__ = ["train_warmstart"]
