"""Split-then-combine generators: input has two halves separated by a divider,
output = OP(left, right) recolored.

Supports:
- horizontal split (rows split, output is iw wide, oh = (ih-1)/2)
- vertical split (cols split, output is ih tall, ow = (iw-1)/2)
- operations: AND, OR, XOR (of non-zero masks)
- output color: specified
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from generators.base import INPUT_NAME, OUTPUT_NAME, make_const, make_int_const, make_model
from pipeline.loader import HEIGHT, WIDTH, CHANNELS


def build_split_combine(axis: str, half_size: int, divider_size: int, op: str,
                         output_color: int, out_h: int, out_w: int) -> onnx.ModelProto:
    """
    axis: 'vertical' (split along cols) or 'horizontal' (split along rows)
    half_size: size of each half along split axis
    divider_size: width/height of the divider (usually 1)
    op: 'and', 'or', 'xor'
    output_color: color for output non-zero cells
    out_h, out_w: fixed output shape
    """
    nodes = []
    inits = []

    # Compute non-zero mask (channels 1..9)
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

    # Shift second half to start of output region
    shift_axis = 3 if axis == 'vertical' else 2
    shift_len = WIDTH if axis == 'vertical' else HEIGHT
    shift_amount = half_size + divider_size
    # idx[i] = i + shift_amount for i in 0..out_size-1, else 0
    shift_idx = [i + shift_amount for i in range(out_w if axis == 'vertical' else out_h)]
    shift_idx += [0] * (shift_len - len(shift_idx))
    inits.append(make_int_const("shift_idx", shift_idx))
    nodes.append(helper.make_node("Gather", ["nz_bin", "shift_idx"], ["shifted"],
                                   axis=shift_axis, name="gather_shift"))

    # Combine: AND, OR, or XOR
    if op == 'and':
        nodes.append(helper.make_node("Mul", ["nz_bin", "shifted"], ["combined"], name="op_and"))
    elif op == 'or':
        nodes.append(helper.make_node("Max", ["nz_bin", "shifted"], ["combined"], name="op_or"))
    elif op == 'xor':
        # XOR = (A != B) = |A - B| for binary
        nodes.append(helper.make_node("Sub", ["nz_bin", "shifted"], ["diff_xor"], name="xor_sub"))
        nodes.append(helper.make_node("Abs", ["diff_xor"], ["combined"], name="xor_abs"))
    else:
        raise ValueError(op)

    # Mask to output region [0:out_h, 0:out_w]
    out_region = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    out_region[0, 0, :out_h, :out_w] = 1.0
    inits.append(make_const("out_region", out_region))
    nodes.append(helper.make_node("Mul", ["combined", "out_region"], ["final_mask"], name="mask_region"))

    # Build output: at final_mask=1 fire channel output_color; elsewhere in output region, fire channel 0
    nodes.append(helper.make_node("Sub", ["out_region", "final_mask"], ["bg_region"], name="bg"))

    ch_col = np.zeros((CHANNELS, 1, 1, 1), dtype=np.float32)
    ch_col[output_color, 0, 0, 0] = 2.0
    inits.append(make_const("ch_col", ch_col))
    nodes.append(helper.make_node("Conv", ["final_mask", "ch_col"], ["col_overlay"],
                                   kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], name="col_conv"))

    ch0_unit = np.zeros((CHANNELS, 1, 1, 1), dtype=np.float32)
    ch0_unit[0, 0, 0, 0] = 2.0
    inits.append(make_const("ch0_unit", ch0_unit))
    nodes.append(helper.make_node("Conv", ["bg_region", "ch0_unit"], ["ch0_overlay"],
                                   kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], name="ch0_conv"))

    nodes.append(helper.make_node("Add", ["col_overlay", "ch0_overlay"], ["combined_out"], name="combine_out"))

    neg_half = make_const("neg_half", np.array([-0.5], dtype=np.float32))
    inits.append(neg_half)
    nodes.append(helper.make_node("Add", ["combined_out", "neg_half"], [OUTPUT_NAME], name="bias"))

    return make_model(nodes, inits, doc=f"split/combine: {axis} {op} color={output_color}")


__all__ = ["build_split_combine"]
