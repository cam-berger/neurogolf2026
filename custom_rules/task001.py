"""Task 001: Kronecker fractal. output[R, C] = input[R%3, C%3] if input[R//3, C//3] != 0 else 0.

Fixed input shape 3x3, output shape 9x9.

Implementation:
1. inner_tiled: Gather input along axis 2 with [0,1,2]*3+zeros to get rows 0..8 cycle 3x
   Then Gather axis 3 with same pattern. Outside (R,C) in [0:9, 0:9], value is garbage.
2. outer_mask: compute nonzero mask of input, then Gather with [0,0,0,1,1,1,2,2,2]+zeros
   Each input mask cell is replicated 3x3.
3. output_region_mask: 1 at [0:9, 0:9], 0 elsewhere.
4. active_mask = outer_mask * output_region_mask
5. background_mask = output_region_mask - active_mask (1 at cells where color should be 0)
6. output = inner_tiled * active_mask + (channel_0_unit) * background_mask
   (scaled to fire above 0 threshold for channel 0)
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from generators.base import INPUT_NAME, OUTPUT_NAME, make_const, make_int_const, make_model
from pipeline.loader import HEIGHT, WIDTH, CHANNELS


def build() -> onnx.ModelProto:
    nodes = []
    inits = []

    # 1. Inner tiling: output[R, C] = input[R % 3, C % 3]
    inner_row_idx = [r % 3 for r in range(9)] + [0] * (HEIGHT - 9)
    inner_col_idx = [c % 3 for c in range(9)] + [0] * (WIDTH - 9)
    inits.append(make_int_const("inner_r_idx", inner_row_idx))
    inits.append(make_int_const("inner_c_idx", inner_col_idx))
    nodes.append(helper.make_node("Gather", [INPUT_NAME, "inner_r_idx"], ["inner_tiled_r"],
                                   axis=2, name="inner_gr"))
    nodes.append(helper.make_node("Gather", ["inner_tiled_r", "inner_c_idx"], ["inner_tiled"],
                                   axis=3, name="inner_gc"))

    # 2. Outer mask expanded (each input cell replicated 3x3)
    # First: nonzero mask [1, 1, 30, 30] (sum channels 1..9 > 0)
    starts_ch = make_int_const("s_ch", [1])
    ends_ch = make_int_const("e_ch", [CHANNELS])
    axes_ch = make_int_const("a_ch", [1])
    steps_ch = make_int_const("st_ch", [1])
    inits.extend([starts_ch, ends_ch, axes_ch, steps_ch])
    nodes.append(helper.make_node("Slice", [INPUT_NAME, "s_ch", "e_ch", "a_ch", "st_ch"],
                                   ["nz_channels"], name="slice_nz"))
    nodes.append(helper.make_node("ReduceMax", ["nz_channels"], ["nz_mask"],
                                   axes=[1], keepdims=1, name="nz_max"))
    nodes.append(helper.make_node("Clip", ["nz_mask"], ["nz_mask_b"],
                                   min=0.0, max=1.0, name="clip_nz"))

    # Replicate each input row 3x (each input cell 3x along row axis)
    outer_row_idx = [r // 3 for r in range(9)] + [0] * (HEIGHT - 9)
    outer_col_idx = [c // 3 for c in range(9)] + [0] * (WIDTH - 9)
    inits.append(make_int_const("outer_r_idx", outer_row_idx))
    inits.append(make_int_const("outer_c_idx", outer_col_idx))
    nodes.append(helper.make_node("Gather", ["nz_mask_b", "outer_r_idx"], ["outer_mask_r"],
                                   axis=2, name="outer_gr"))
    nodes.append(helper.make_node("Gather", ["outer_mask_r", "outer_c_idx"], ["outer_mask"],
                                   axis=3, name="outer_gc"))

    # 3. Output region mask: 1 at [0:9, 0:9]
    output_region = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    output_region[0, 0, :9, :9] = 1.0
    inits.append(make_const("out_region", output_region))

    # 4. active_mask = outer_mask * output_region
    nodes.append(helper.make_node("Mul", ["outer_mask", "out_region"], ["active_mask"], name="am_mul"))

    # 5. background_mask = output_region - active_mask
    nodes.append(helper.make_node("Sub", ["out_region", "active_mask"], ["bg_mask"], name="bgm_sub"))

    # 6. Compose output:
    # output_colored = inner_tiled * active_mask (broadcast across 10 channels)
    nodes.append(helper.make_node("Mul", ["inner_tiled", "active_mask"], ["tiled_active"], name="ta_mul"))

    # background at channel 0: bg_mask * channel_0_indicator
    # channel_0_indicator is [1, 10, 1, 1] with 1 at channel 0, 0 elsewhere
    ch0 = np.zeros((1, CHANNELS, 1, 1), dtype=np.float32)
    ch0[0, 0, 0, 0] = 1.0
    inits.append(make_const("ch0_unit", ch0))
    nodes.append(helper.make_node("Mul", ["ch0_unit", "bg_mask"], ["bg_ch0"], name="bg_ch0"))

    # Scale both to fire above threshold 0.5
    # For tiled_active: values are 0 or 1 (one-hot). Multiply by 2 and shift by -0.5 via Add.
    # Simplest: add both, then multiply by 2 and subtract 0.5.
    nodes.append(helper.make_node("Add", ["tiled_active", "bg_ch0"], ["combined"], name="combine"))
    two_c = make_const("two_c", np.array([2.0], dtype=np.float32))
    neg_half_c = make_const("neg_half_c", np.array([-0.5], dtype=np.float32))
    inits.extend([two_c, neg_half_c])
    nodes.append(helper.make_node("Mul", ["combined", "two_c"], ["scaled"], name="sc_mul"))
    nodes.append(helper.make_node("Add", ["scaled", "neg_half_c"], [OUTPUT_NAME], name="final_bias"))

    return make_model(nodes, inits, doc="task 001: kronecker fractal 3x3 -> 9x9")


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return build()


__all__ = ["build", "generate"]
