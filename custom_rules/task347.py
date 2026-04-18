"""Task 347: input (3, 6) with two 3x3 halves. Output = OR of halves, recolored to 6.

Rule: output[r, c] = 6 if input[r, c] != 0 OR input[r, c+3] != 0 else 0.
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

    # Extract any-non-zero mask (ReduceMax over channels 1..9) -> [1, 1, 30, 30]
    s_ch = make_int_const("s_ch", [1])
    e_ch = make_int_const("e_ch", [CHANNELS])
    a_ch = make_int_const("a_ch", [1])
    st_ch = make_int_const("st_ch", [1])
    inits.extend([s_ch, e_ch, a_ch, st_ch])
    nodes.append(helper.make_node("Slice", [INPUT_NAME, "s_ch", "e_ch", "a_ch", "st_ch"],
                                   ["nz_ch"], name="slice_nz"))
    nodes.append(helper.make_node("ReduceMax", ["nz_ch"], ["nz_mask"],
                                   axes=[1], keepdims=1, name="nz_max"))
    nodes.append(helper.make_node("Clip", ["nz_mask"], ["nz_bin"], min=0.0, max=1.0, name="clip_nz"))

    # Left half (cols 0..2) and right half (cols 3..5) — shift right half to the left via Gather
    # Right half: we gather cols [3, 4, 5, 0, 1, ..., 29] on axis 3, then take cols 0..2.
    # Simpler: shift cols [3, 4, 5] to [0, 1, 2] — use Gather with indices [3, 4, 5, 3, 4, 5, ..., 3] (any padding is fine for cols beyond).
    right_idx = [3, 4, 5] + [0] * (WIDTH - 3)
    inits.append(make_int_const("right_idx", right_idx))
    nodes.append(helper.make_node("Gather", ["nz_bin", "right_idx"], ["right_shifted"],
                                   axis=3, name="gather_right"))

    # OR: max of left (nz_bin as-is) and right_shifted
    nodes.append(helper.make_node("Max", ["nz_bin", "right_shifted"], ["or_mask"], name="max_or"))

    # Mask to the output region [0:3, 0:3]
    out_region = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    out_region[0, 0, :3, :3] = 1.0
    inits.append(make_const("out_region", out_region))
    nodes.append(helper.make_node("Mul", ["or_mask", "out_region"], ["final_mask"], name="mask_region"))

    # Build output: at final_mask=1, channel 6 fires; at (in output region but final_mask=0), channel 0 fires; outside region, nothing.
    # Identity base for channel 0 = 1 at output region (bias -0.5, input 0 → -0.5 neg)
    # Simpler: create channel 0 at (output_region - final_mask), channel 6 at final_mask
    nodes.append(helper.make_node("Sub", ["out_region", "final_mask"], ["bg_region"], name="bg"))

    # Channel 6 firing at final_mask: create [1, 10, 30, 30] overlay
    ch6_unit = np.zeros((CHANNELS, 1, 1, 1), dtype=np.float32)
    ch6_unit[6, 0, 0, 0] = 2.0  # 2 * 1 = 2 > 0 fires
    inits.append(make_const("ch6_unit", ch6_unit))
    nodes.append(helper.make_node("Conv", ["final_mask", "ch6_unit"], ["ch6_overlay"],
                                   kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], name="ch6_conv"))

    ch0_unit = np.zeros((CHANNELS, 1, 1, 1), dtype=np.float32)
    ch0_unit[0, 0, 0, 0] = 2.0
    inits.append(make_const("ch0_unit", ch0_unit))
    nodes.append(helper.make_node("Conv", ["bg_region", "ch0_unit"], ["ch0_overlay"],
                                   kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], name="ch0_conv"))

    # Combine and bias
    nodes.append(helper.make_node("Add", ["ch6_overlay", "ch0_overlay"], ["combined"], name="combine"))

    # Bias -0.5 per channel (but only where combined is 0, it'll decode to "no color"; where 2, it'll fire)
    # Actually we want:
    #   - combined = 2 at channel 6 & final_mask=1: output = 2 - 0.5 = 1.5 > 0 fires
    #   - combined = 2 at channel 0 & bg_region=1: output = 1.5 fires
    #   - elsewhere (outside region): combined = 0, output = -0.5 no fire
    neg_half = make_const("neg_half", np.array([-0.5], dtype=np.float32))
    inits.append(neg_half)
    nodes.append(helper.make_node("Add", ["combined", "neg_half"], [OUTPUT_NAME], name="bias"))

    return make_model(nodes, inits, doc="task 347: OR of two 3x3 halves, recolor to 6")


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return build()


__all__ = ["build", "generate"]
