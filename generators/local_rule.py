"""Local-rule generator: Conv(kxk, pad=k//2) -> ReLU -> Conv(1x1).

Both convs have no bias so that "outside-grid" cells (all-zero channels
across the stack) remain exactly zero all the way through — this matches
the scoring semantics where output cells with no active channel decode to
'no color' and get trimmed.

Training pools every visible (train + test + arc_gen) pair: those are all
present in the task JSON and the network must generalize to a private
holdout, so pooling is legitimate and gives vastly more windows to fit.

After training, the generator runs the net against every pair with the
official threshold-at-zero decision rule; it returns None if any pair is
wrong rather than shipping an ONNX that will fail scoring.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import onnx

from generators.base import NetworkGenerator
from pipeline.loader import CHANNELS, GRID_SHAPE, encode_grid, get_all_pairs

_N_STEPS = 5000
_LR = 5e-3
_L1 = 1e-5
_OPSET = 10
_TRAIN_SAMPLE = 50  # cap training batch; verification still runs on all pairs
_EVAL_EVERY = 100
_MARGIN = 0.5  # hinge loss margin — logits must sit ±_MARGIN from the scoring threshold at 0


def _torch():
    import torch  # lazy
    import torch.nn as nn
    return torch, nn


def _build_stack(nn, kernel: int):
    pad = kernel // 2
    return nn.Sequential(
        nn.Conv2d(CHANNELS, CHANNELS, kernel_size=kernel, padding=pad, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(CHANNELS, CHANNELS, kernel_size=1, bias=False),
    )


def _encode_pairs(pairs: list) -> tuple[np.ndarray, np.ndarray]:
    xs = np.concatenate([encode_grid(inp) for inp, _ in pairs], axis=0)
    ys = np.concatenate([encode_grid(out) for _, out in pairs], axis=0)
    return xs, ys


def _pick_training_subset(task: dict) -> list | None:
    """Always include every train pair; sample up to _TRAIN_SAMPLE total from the pool."""
    pool = get_all_pairs(task)
    if not pool:
        return None
    train_n = len(task.get("train", []))
    extras = pool[train_n:]
    if len(pool) <= _TRAIN_SAMPLE or not extras:
        return pool
    rng = np.random.default_rng(0)
    extra_idx = rng.choice(len(extras), size=_TRAIN_SAMPLE - train_n, replace=False)
    return pool[:train_n] + [extras[i] for i in extra_idx]


class LocalRuleGenerator(NetworkGenerator):
    def __init__(self, kernel: int = 3):
        self.kernel = kernel
        self.family = f"local_rule_{kernel}x{kernel}"

    def can_generate(self, task: dict, features: dict) -> bool:
        if not features.get("output_shape_eq_input"):
            return False
        k = features.get("max_local_context_needed", 99)
        return k <= self.kernel

    def generate(self, task: dict, features: dict) -> onnx.ModelProto | None:
        # Hand-coded rules take precedence over training. The registry is
        # keyed by task id; we pick it up from features when available, or
        # fall back to the training loop if the task isn't registered.
        task_id = features.get("_task_id") if isinstance(features, dict) else None
        if task_id is not None:
            from custom_rules import get as _get_custom
            fn = _get_custom(int(task_id))
            if fn is not None:
                return fn(task, features)

        subset = _pick_training_subset(task)
        if subset is None:
            return None
        all_pairs = get_all_pairs(task)

        torch, nn = _torch()
        xs_np, ys_np = _encode_pairs(subset)
        all_xs_np, all_ys_np = _encode_pairs(all_pairs)
        xs = torch.from_numpy(xs_np).float()
        ys = torch.from_numpy(ys_np).float()
        all_xs = torch.from_numpy(all_xs_np).float()
        all_ys = torch.from_numpy(all_ys_np).float()

        model = _build_stack(nn, self.kernel)
        opt = torch.optim.Adam(model.parameters(), lr=_LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_N_STEPS)

        # Signed target: +1 where a channel should fire, -1 elsewhere. The
        # hinge loss max(0, margin - sign*logit) has zero gradient for any
        # cell already beyond the margin, so optimization effort concentrates
        # on the cells still on the wrong side of the scoring threshold.
        sign = (2.0 * ys - 1.0)

        best_errors = None
        best_state = None

        for step in range(_N_STEPS):
            opt.zero_grad()
            logits = model(xs)
            hinge = torch.clamp(_MARGIN - sign * logits, min=0.0)
            loss = hinge.mean()
            l1 = sum(p.abs().sum() for p in model.parameters())
            (loss + _L1 * l1).backward()
            opt.step()
            sched.step()

            if step % _EVAL_EVERY == 0 or step == _N_STEPS - 1:
                with torch.no_grad():
                    # Evaluate on the FULL pool — that is what we have to
                    # beat, and it reveals overfitting on the sampled subset.
                    preds = (model(all_xs) > 0.0).float()
                    errors = int((preds != all_ys).any(dim=(1, 2, 3)).sum().item())
                if best_errors is None or errors < best_errors:
                    best_errors = errors
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if errors == 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Final verification: every pair in the full pool must match at >0.
        with torch.no_grad():
            final = (model(all_xs).numpy() > 0.0).astype(np.float32)
        if not np.array_equal(final, all_ys_np):
            return None

        # Export opset-10 ONNX via a temp file, then reload as ModelProto.
        dummy = torch.zeros(GRID_SHAPE, dtype=torch.float32)
        with NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            torch.onnx.export(
                model,
                dummy,
                tmp_path,
                input_names=["input"],
                output_names=["output"],
                opset_version=_OPSET,
                do_constant_folding=True,
            )
            proto = onnx.load(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return proto


__all__ = ["LocalRuleGenerator"]
