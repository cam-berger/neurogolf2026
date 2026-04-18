"""Connected-component labeling primitive (ONNX-compatible, opset 10).

Given input [1, 10, 30, 30] one-hot encoded grid, computes a labels tensor
[1, 1, 30, 30] where each same-color connected region has a unique positive
integer label (encoded as float). Background cells are 0.

Algorithm (no Loop op):
1. For each color c in 1..9:
   - mask_c = input[:, c, :, :]  (binary)
   - label_c initialized as (r*W + c + 1) at mask_c=1, LARGE at mask_c=0
   - Iterate min-propagation 30 times: new_label = min over 3x3 neighborhood
     - Min-pool implemented as -MaxPool(-x)
     - LARGE at background ensures background doesn't pull valid labels down
   - Final label_c = min_pool_result * mask_c (zeros out background)
   - Shift labels to per-color range: label_c += (c-1) * 900 at mask positions

2. Combine: labels = sum over c of label_c
   (Each cell has exactly one color, so only one stream contributes non-zero.)

Total iterations across 9 colors = 9 * 30 = 270 MaxPool ops. Each op is tiny
(no params), so file size is dominated by initializers (index offsets etc).
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from generators.base import INPUT_NAME, OUTPUT_NAME, make_const, make_int_const, make_model
from pipeline.loader import HEIGHT, WIDTH, CHANNELS


LARGE = 100000.0  # sentinel for background — must exceed any valid label
N_ITER = 30       # enough for any 30x30 grid (max Manhattan distance 58 → 30 iterations of 3x3 covers it)


def _slice_channel(input_name: str, ch: int, tag: str):
    """Extract one channel as [1, 1, H, W]."""
    nodes = []
    inits = []
    starts = make_int_const(f"s_{tag}", [ch])
    ends = make_int_const(f"e_{tag}", [ch + 1])
    axes = make_int_const(f"a_{tag}", [1])
    steps = make_int_const(f"st_{tag}", [1])
    inits.extend([starts, ends, axes, steps])
    nodes.append(helper.make_node(
        "Slice",
        [input_name, f"s_{tag}", f"e_{tag}", f"a_{tag}", f"st_{tag}"],
        [f"{tag}_mask"], name=f"slice_{tag}",
    ))
    return nodes, inits, f"{tag}_mask"


def _build_per_color_cc(mask_name: str, color: int, color_offset: float):
    """Build CC labeling for one color channel.

    Returns (nodes, inits, final_label_tensor_name).

    final_label_tensor at (r, c) is:
      - offset + min_label_in_component if mask=1 at (r, c)
      - 0 if mask=0
    """
    tag = f"c{color}"
    nodes = []
    inits = []

    # Initialize: label = (r*W + c + 1) at mask=1, LARGE at mask=0.
    # Precompute base = [1..HEIGHT*WIDTH] reshaped to [1, 1, H, W]
    base = np.arange(1, HEIGHT * WIDTH + 1, dtype=np.float32).reshape(1, 1, HEIGHT, WIDTH)
    base_const = make_const(f"base_{tag}", base)
    inits.append(base_const)
    large_const = make_const(f"large_{tag}", np.full((1, 1, HEIGHT, WIDTH), LARGE, dtype=np.float32))
    inits.append(large_const)

    # label_init = base * mask + LARGE * (1 - mask)
    #            = base * mask + LARGE - LARGE * mask
    #            = (base - LARGE) * mask + LARGE
    diff_const = make_const(f"diff_{tag}", np.full((1,), -LARGE, dtype=np.float32))
    inits.append(diff_const)
    # base_minus_large = base + diff (broadcast)
    nodes.append(helper.make_node("Add", [f"base_{tag}", f"diff_{tag}"], [f"bml_{tag}"],
                                   name=f"bml_{tag}"))
    # (base-LARGE) * mask
    nodes.append(helper.make_node("Mul", [f"bml_{tag}", mask_name], [f"bmlm_{tag}"],
                                   name=f"bmlm_{tag}"))
    # + LARGE
    nodes.append(helper.make_node("Add", [f"bmlm_{tag}", f"large_{tag}"], [f"init_{tag}"],
                                   name=f"init_{tag}"))

    # Iterative min-propagation via -MaxPool(-x)
    # neg_const for negation (just Neg op)
    cur = f"init_{tag}"
    for step in range(N_ITER):
        # neg = -cur
        neg = f"neg_{tag}_{step}"
        nodes.append(helper.make_node("Neg", [cur], [neg], name=f"neg_{tag}_{step}"))
        # maxpool 3x3 pad=1
        pool = f"maxp_{tag}_{step}"
        nodes.append(helper.make_node(
            "MaxPool", [neg], [pool],
            kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1],
            name=f"mp_{tag}_{step}",
        ))
        # re-negate
        new_cur = f"lbl_{tag}_{step+1}"
        nodes.append(helper.make_node("Neg", [pool], [new_cur], name=f"negb_{tag}_{step}"))

        # Clamp to mask: any cell that wasn't mask=1 should stay LARGE
        # Actually: after pooling, a background cell (originally LARGE) could now have a smaller value if a foreground neighbor propagated. We want background cells to remain LARGE throughout iteration (they don't participate).
        # Formula: masked = new_cur * mask + LARGE * (1 - mask)
        #        = (new_cur - LARGE) * mask + LARGE
        offset_name = f"off_{tag}_{step+1}"
        nodes.append(helper.make_node("Sub", [new_cur, f"large_{tag}"], [f"offs_{tag}_{step+1}"],
                                       name=f"sub_off_{tag}_{step+1}"))
        nodes.append(helper.make_node("Mul", [f"offs_{tag}_{step+1}", mask_name],
                                       [f"offsm_{tag}_{step+1}"], name=f"mul_offm_{tag}_{step+1}"))
        nodes.append(helper.make_node("Add", [f"offsm_{tag}_{step+1}", f"large_{tag}"],
                                       [f"clamped_{tag}_{step+1}"], name=f"add_clamp_{tag}_{step+1}"))
        cur = f"clamped_{tag}_{step+1}"

    # After iterations: `cur` has min labels at mask=1, LARGE at mask=0.
    # Zero out background: final = (cur - LARGE) * mask = label_or_0
    # Actually wait: at mask=0 cells, cur = LARGE. We want 0. So final = (cur * mask).
    # But at mask=1 cells, cur * mask = cur (since mask=1).
    # At mask=0 cells, cur * mask = LARGE * 0 = 0.
    nodes.append(helper.make_node("Mul", [cur, mask_name], [f"label_unshifted_{tag}"],
                                   name=f"unshift_{tag}"))

    # Shift labels by color_offset to distinguish colors.
    # Only shift at mask=1 cells: shifted = label_unshifted + color_offset * mask
    if color_offset != 0.0:
        off_c = make_const(f"coloff_{tag}", np.array([color_offset], dtype=np.float32))
        inits.append(off_c)
        nodes.append(helper.make_node("Mul", [mask_name, f"coloff_{tag}"], [f"off_term_{tag}"],
                                       name=f"off_term_{tag}"))
        nodes.append(helper.make_node("Add", [f"label_unshifted_{tag}", f"off_term_{tag}"],
                                       [f"label_{tag}"], name=f"final_label_{tag}"))
    else:
        nodes.append(helper.make_node("Identity", [f"label_unshifted_{tag}"], [f"label_{tag}"],
                                       name=f"ident_{tag}"))

    return nodes, inits, f"label_{tag}"


def build_cc_labels() -> onnx.ModelProto:
    """Build a standalone ONNX model that computes CC labels.

    Input: [1, 10, 30, 30]
    Output: [1, 1, 30, 30] — CC labels (0 = background, positive floats = unique labels).
    """
    nodes = []
    inits = []

    # Slice each color channel 1..9 and compute per-color CC labels
    label_names = []
    for color in range(1, 10):
        tag = f"c{color}"
        slice_nodes, slice_inits, mask_name = _slice_channel(INPUT_NAME, color, tag)
        nodes += slice_nodes
        inits += slice_inits

        # color_offset so labels from different colors don't collide
        offset = float((color - 1) * HEIGHT * WIDTH)  # 0, 900, 1800, ..., 7200
        cc_nodes, cc_inits, label_name = _build_per_color_cc(mask_name, color, offset)
        nodes += cc_nodes
        inits += cc_inits
        label_names.append(label_name)

    # Combine via successive Add (sum across colors — only one is non-zero per cell)
    cur = label_names[0]
    for i, name in enumerate(label_names[1:], 1):
        out = OUTPUT_NAME if i == len(label_names) - 1 else f"combined_{i}"
        nodes.append(helper.make_node("Add", [cur, name], [out], name=f"combine_{i}"))
        cur = out

    return make_model(nodes, inits, doc=f"connected-components labeling, {N_ITER} iterations")


__all__ = ["build_cc_labels", "N_ITER", "LARGE"]
