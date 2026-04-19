"""Task 007: period-3 cycle fill.

For a fixed 7x7 grid, cells cycle through 3 colors based on their flat index mod 3.
Non-zero input cells determine which color belongs to each residue class.
Output fills the entire 7x7 grid according to that mapping.

Rule: output[r, c] = color_of_class[(r*7 + c) % 3]
where color_of_class[k] = any input color at a cell whose class == k.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from generators.base import INPUT_NAME, OUTPUT_NAME, make_const, make_int_const, make_model
from pipeline.loader import HEIGHT, WIDTH, CHANNELS


GRID_SIZE = 7


def build() -> onnx.ModelProto:
    nodes = []
    inits = []

    # Precompute 3 class masks: 1 at cells in 7x7 area where (r*7+c) % 3 == k, else 0
    for k in range(3):
        mask = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if (r * GRID_SIZE + c) % 3 == k:
                    mask[0, 0, r, c] = 1.0
        inits.append(make_const(f"cm_{k}", mask))

    # Slice channels 1..9 of input (skip background channel 0)
    s_ch = make_int_const("s_ch", [1])
    e_ch = make_int_const("e_ch", [CHANNELS])
    a_ch = make_int_const("a_ch", [1])
    st_ch = make_int_const("st_ch", [1])
    inits.extend([s_ch, e_ch, a_ch, st_ch])
    nodes.append(helper.make_node("Slice", [INPUT_NAME, "s_ch", "e_ch", "a_ch", "st_ch"],
                                   ["colors_1_9"], name="slice_c19"))

    # For each class k: find which colors are present at its cells.
    #   masked_k = colors_1_9 * cm_k (broadcast)
    #   has_color_k [1, 9, 1, 1] = ReduceMax(masked_k, axes=[2, 3])
    contribs = []
    for k in range(3):
        nodes.append(helper.make_node("Mul", ["colors_1_9", f"cm_{k}"], [f"masked_{k}"],
                                       name=f"mul_cm_{k}"))
        nodes.append(helper.make_node("ReduceMax", [f"masked_{k}"], [f"has_c_{k}"],
                                       axes=[2, 3], keepdims=1, name=f"redmax_{k}"))
        # has_c_k [1, 9, 1, 1] × cm_k [1, 1, 30, 30] → [1, 9, 30, 30]
        nodes.append(helper.make_node("Mul", [f"has_c_{k}", f"cm_{k}"], [f"contrib_{k}"],
                                       name=f"contrib_{k}"))
        contribs.append(f"contrib_{k}")

    # Sum class contributions → [1, 9, 30, 30]
    nodes.append(helper.make_node("Add", [contribs[0], contribs[1]], ["sum_01"], name="sum_01"))
    nodes.append(helper.make_node("Add", ["sum_01", contribs[2]], ["channels_1_9"], name="sum_012"))

    # Prepend channel 0 = 0 everywhere to get [1, 10, 30, 30]
    ch0_zero = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    inits.append(make_const("ch0_zero", ch0_zero))
    nodes.append(helper.make_node("Concat", ["ch0_zero", "channels_1_9"], ["combined"],
                                   axis=1, name="concat_ch"))

    # Scale and bias: values 0 or 1. 1 * 2 - 0.5 = 1.5 > 0 (fire), 0 * 2 - 0.5 = -0.5 (no fire).
    two_c = make_const("two_c", np.array([2.0], dtype=np.float32))
    neg_half = make_const("neg_half", np.array([-0.5], dtype=np.float32))
    inits.extend([two_c, neg_half])
    nodes.append(helper.make_node("Mul", ["combined", "two_c"], ["scaled"], name="sc"))
    nodes.append(helper.make_node("Add", ["scaled", "neg_half"], [OUTPUT_NAME], name="bias"))

    return make_model(nodes, inits, doc="task 007: period-3 flat-index cycle fill on 7x7")


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return build()


__all__ = ["build", "generate"]
