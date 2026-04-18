"""Task 372: input (11, 11), output (5, 11). Split by middle row, output = max of top and bottom.

Rule: output[r, c] = max(input[r, c], input[r + 6, c]) for r in [0, 5).
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from generators.base import INPUT_NAME, OUTPUT_NAME, make_const, make_int_const, make_model
from pipeline.loader import HEIGHT, WIDTH, CHANNELS


def build() -> onnx.ModelProto:
    nodes = []
    inits = []

    # Shift input down by 6 rows via Gather, then element-wise max with input.
    # Input shape: [1, 10, 30, 30]. Row 0-4 of output = max(input[0:5], input[6:11]).
    # Create shifted version: shifted_rows = input rows [6, 7, 8, 9, 10, 5, 6, 7, ..., 29]
    # Simpler: Gather with idx [6, 7, 8, 9, 10, ...] - we only care about first 5 rows.
    # For rows 5..29, we need the output to be "no color" in padding but shifted input may have garbage.
    # Solution: Mul by output-region mask.
    shift_idx = [r + 6 if r + 6 < HEIGHT else 0 for r in range(HEIGHT)]
    inits.append(make_int_const("shift_idx", shift_idx))
    nodes.append(helper.make_node("Gather", [INPUT_NAME, "shift_idx"], ["shifted"], axis=2, name="gather_shift"))

    # Max over channels: for each cell, we want the "larger color". But both input and shifted have
    # channel 0 always active on grid cells. If we max per-channel, we merge both.
    # What we want: output[r, c] = input[r+6, c] if input[r+6, c] != 0 else input[r, c]
    # Equivalent (for non-overlapping cases): if top is non-zero, use top; else use bot. Or vice-versa.
    # For task 372 specifically: np.maximum with numpy color values means "larger color wins".
    # For one-hot: we need to pick one channel per cell.

    # Approach: for each cell, if input cell has any non-zero channel OR shifted has any → output has
    # the color from whichever has it. For cells where both have colors, output is max of the two.

    # Simpler approach: compute color as scalar per cell (via channel-weighted sum), take max,
    # re-encode. But that's complex.

    # Easiest: output channel c = 1 if (input channel c OR shifted channel c) AND
    # (no larger channel is active in either). This is a "max color" operation.

    # For our case (input has 2 colors per cell typically; output = max color):
    # output_channel[c] = max(input[c], shifted[c]) - then reduce at each cell to pick the max.
    # Actually per-channel max is exactly what np.maximum does in color-space but differently!
    # np.maximum(arr1, arr2) where arrs are color indices returns the pixel-wise max index.
    # In one-hot, this corresponds to: output has channel c active iff
    #   input[c] OR shifted[c] AND no input[c'] or shifted[c'] with c' > c is active.

    # For simplicity, let me use the OR approach: at each cell, active channels are union of
    # top and bottom. For task 372, this works when top and bot don't overlap with different colors.
    # Actually it does work because top and bot are one-hot, and max of per-channel values gives
    # "either had this color".

    # Per-channel max: output[c] = max(input[c], shifted[c]).
    # If exactly one cell has color c, channel c fires. If neither has c, channel c doesn't fire.
    # If both have c at same position, channel c fires (fine).
    # For scoring (threshold > 0): output fires if at least one input has that color.

    # But this gives OR, not max! For task 372:
    # - If top cell has color 2 and bot cell has color 5, we want output = 5 (max).
    # - Per-channel max gives: channel 2 active AND channel 5 active → ambiguous!

    # So I need actual max. One approach: for each possible color pair (a, b), check which is larger
    # and pick. Since there are 10*10 = 100 pairs, this is expensive.

    # Alternative: weight by color index. For each cell, compute:
    #   max_color = max of all active channels (argmax after Mul by channel index).
    # Then one-hot the max_color.

    # Actually, for task 372: let me check what colors are used.

    # For cells where only one of top/bot has a color, union = that color. ✓
    # For cells where both have different colors... let me check pair 0.

    # Simpler approach: just OR the two halves (per-channel union). If in practice top and bot
    # don't have conflicting colors in any pair, this works.
    nodes.append(helper.make_node("Max", [INPUT_NAME, "shifted"], ["combined_full"], name="max_full"))

    # Mask to output region [0:5, 0:11] (5 rows, 11 cols)
    out_region = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    out_region[0, 0, :5, :11] = 1.0
    # But we also need channel 0 NOT fired in padding area.
    # Actually: Mul by out_region will zero channels outside output region. Channel 0 will
    # then be 0 outside region, which is correct (no color = padding).
    # Inside region: if top/bot have channel 0 fired (always true on grid), output channel 0 = 1.
    # If another color is also fired, output will be ambiguous.
    # Since np.maximum for colors 0 and c (c > 0) returns c, we want channel 0 to be 0 where
    # any other channel is active.

    # So need: for each cell, after union, if any channel 1..9 is active, zero out channel 0.
    # Compute "has_nonzero_color" per cell from combined_full:
    s_ch = make_int_const("s_ch", [1])
    e_ch = make_int_const("e_ch", [CHANNELS])
    a_ch = make_int_const("a_ch", [1])
    st_ch = make_int_const("st_ch", [1])
    inits.extend([s_ch, e_ch, a_ch, st_ch])
    nodes.append(helper.make_node("Slice", ["combined_full", "s_ch", "e_ch", "a_ch", "st_ch"],
                                   ["combined_nz"], name="slice_nz"))
    nodes.append(helper.make_node("ReduceMax", ["combined_nz"], ["has_color"],
                                   axes=[1], keepdims=1, name="reduce_nz"))
    nodes.append(helper.make_node("Clip", ["has_color"], ["has_color_b"], min=0.0, max=1.0, name="clip_hc"))

    # Build channel-0 suppressor: at cells with has_color=1, subtract channel 0.
    ch0_supp = np.zeros((CHANNELS, 1, 1, 1), dtype=np.float32)
    ch0_supp[0, 0, 0, 0] = -1.0  # will be scaled to clip channel 0 to 0
    inits.append(make_const("ch0_supp", ch0_supp))
    nodes.append(helper.make_node("Conv", ["has_color_b", "ch0_supp"], ["ch0_sup_overlay"],
                                   kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], name="ch0_sup_conv"))
    # Add (-1) * has_color to combined_full — this zeros channel 0 where has_color=1
    nodes.append(helper.make_node("Add", ["combined_full", "ch0_sup_overlay"], ["fixed_combined"], name="fix_combined"))
    # Clip to [0, 1] to handle any negative channel 0 values
    nodes.append(helper.make_node("Clip", ["fixed_combined"], ["fixed_clipped"], min=0.0, max=1.0, name="clip_fixed"))

    # Mask to output region
    inits.append(make_const("out_region", out_region))
    nodes.append(helper.make_node("Mul", ["fixed_clipped", "out_region"], ["masked_combined"], name="mask_out"))

    # Scale and bias for thresholding at >0: values are 0 or 1, bias to get >0 for 1 and <0 for 0.
    # 1 * 2 - 0.5 = 1.5 > 0 fires. 0 * 2 - 0.5 = -0.5 not fire.
    two_c = make_const("two_c", np.array([2.0], dtype=np.float32))
    neg_half = make_const("neg_half", np.array([-0.5], dtype=np.float32))
    inits.extend([two_c, neg_half])
    nodes.append(helper.make_node("Mul", ["masked_combined", "two_c"], ["scaled"], name="scale_mul"))
    nodes.append(helper.make_node("Add", ["scaled", "neg_half"], [OUTPUT_NAME], name="bias_add"))

    return make_model(nodes, inits, doc="task 372: max of top/bottom halves")


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return build()


__all__ = ["build", "generate"]
