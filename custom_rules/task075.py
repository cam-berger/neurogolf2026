"""Task 075: template stamping.

Fixed 9x13 grid. Template is input[0:3, 0:3]. Divider column of 5s at col 3.
Right region (col >= 4) has markers (color 1). Each marker gets replaced by
a 3x3 stamp of the template centered on the marker.

ONNX approach:
1. Build marker mask: input channel 1 active AND col >= 4.
2. For each offset (du, dv) in {-1, 0, 1}^2:
   - Shift marker mask by (du, dv): shifted[r, c] = marker_mask[r-du, c-dv]
   - Slice input cell (1+du, 1+dv) → template value for this stamp position
   - Contribution: shifted * template_cell
3. Sum contributions → stamp_contribution [1, 10, 30, 30]
4. any_stamp[r, c] = 1 if any of 9 shifted marker masks hits this cell
5. Output = stamp_contribution + input * (1 - any_stamp)
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from generators.base import INPUT_NAME, OUTPUT_NAME, make_const, make_int_const, make_model
from pipeline.loader import HEIGHT, WIDTH, CHANNELS


def _slice_cell(row: int, col: int, tag: str, inits: list, nodes: list) -> str:
    starts = make_int_const(f"s_{tag}", [0, 0, row, col])
    ends = make_int_const(f"e_{tag}", [1, CHANNELS, row + 1, col + 1])
    axes = make_int_const(f"a_{tag}", [0, 1, 2, 3])
    steps = make_int_const(f"st_{tag}", [1, 1, 1, 1])
    inits.extend([starts, ends, axes, steps])
    nodes.append(helper.make_node("Slice",
        [INPUT_NAME, f"s_{tag}", f"e_{tag}", f"a_{tag}", f"st_{tag}"],
        [f"cell_{tag}"], name=f"slice_{tag}"))
    return f"cell_{tag}"


def _shift_2d(mask_name: str, du: int, dv: int, tag: str, inits: list, nodes: list) -> str:
    """Shift [1, 1, H, W] mask by (du, dv): new[r, c] = old[r-du, c-dv], 0 outside."""
    # Row shift
    row_idx = []
    row_valid = []
    for r in range(HEIGHT):
        src = r - du
        if 0 <= src < HEIGHT:
            row_idx.append(src); row_valid.append(1.0)
        else:
            row_idx.append(0); row_valid.append(0.0)
    inits.append(make_int_const(f"ri_{tag}", row_idx))
    rvm = np.array(row_valid, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    inits.append(make_const(f"rvm_{tag}", rvm))
    nodes.append(helper.make_node("Gather", [mask_name, f"ri_{tag}"], [f"rg_{tag}"],
                                   axis=2, name=f"rg_{tag}"))
    nodes.append(helper.make_node("Mul", [f"rg_{tag}", f"rvm_{tag}"], [f"rgm_{tag}"], name=f"rgm_{tag}"))

    # Col shift
    col_idx = []
    col_valid = []
    for c in range(WIDTH):
        src = c - dv
        if 0 <= src < WIDTH:
            col_idx.append(src); col_valid.append(1.0)
        else:
            col_idx.append(0); col_valid.append(0.0)
    inits.append(make_int_const(f"ci_{tag}", col_idx))
    cvm = np.array(col_valid, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    inits.append(make_const(f"cvm_{tag}", cvm))
    nodes.append(helper.make_node("Gather", [f"rgm_{tag}", f"ci_{tag}"], [f"cg_{tag}"],
                                   axis=3, name=f"cg_{tag}"))
    nodes.append(helper.make_node("Mul", [f"cg_{tag}", f"cvm_{tag}"], [f"shifted_{tag}"], name=f"cgm_{tag}"))
    return f"shifted_{tag}"


def build() -> onnx.ModelProto:
    nodes = []
    inits = []

    # 1. Extract channel 1 from input [1, 10, 30, 30] → [1, 1, 30, 30]
    s_ch = make_int_const("s_ch1", [1])
    e_ch = make_int_const("e_ch1", [2])
    a_ch = make_int_const("a_ch1", [1])
    st_ch = make_int_const("st_ch1", [1])
    inits.extend([s_ch, e_ch, a_ch, st_ch])
    nodes.append(helper.make_node("Slice", [INPUT_NAME, "s_ch1", "e_ch1", "a_ch1", "st_ch1"],
                                   ["ch1"], name="slice_ch1"))

    # 2. Mask by col >= 4 (markers are only in right region)
    col_mask = np.zeros((1, 1, 1, WIDTH), dtype=np.float32)
    col_mask[0, 0, 0, 4:] = 1.0
    inits.append(make_const("col_mask_ge4", col_mask))
    nodes.append(helper.make_node("Mul", ["ch1", "col_mask_ge4"], ["marker_mask"], name="marker_mask"))

    # 3. For each of 9 offsets (du, dv), shift marker_mask and multiply by template cell
    contribs = []
    any_stamps = []
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            tag = f"d{du+1}{dv+1}"
            shifted = _shift_2d("marker_mask", du, dv, tag, inits, nodes)
            # Template cell at (1+du, 1+dv)
            cell = _slice_cell(1 + du, 1 + dv, f"tpl_{tag}", inits, nodes)
            # Broadcast multiply: [1, 1, 30, 30] * [1, 10, 1, 1] = [1, 10, 30, 30]
            contrib = f"contrib_{tag}"
            nodes.append(helper.make_node("Mul", [shifted, cell], [contrib], name=f"contrib_mul_{tag}"))
            contribs.append(contrib)
            any_stamps.append(shifted)

    # 4. stamp_sum = sum of all 9 contributions
    cur = contribs[0]
    for i, c in enumerate(contribs[1:], 1):
        out = f"stamp_sum_{i}" if i < len(contribs) - 1 else "stamp_sum"
        nodes.append(helper.make_node("Add", [cur, c], [out], name=f"stamp_add_{i}"))
        cur = out

    # 5. any_stamp_mask = max of all 9 shifted markers (binary OR)
    cur = any_stamps[0]
    for i, a in enumerate(any_stamps[1:], 1):
        out = f"any_stamp_{i}" if i < len(any_stamps) - 1 else "any_stamp"
        nodes.append(helper.make_node("Max", [cur, a], [out], name=f"any_max_{i}"))
        cur = out

    # 6. input * (1 - any_stamp): preserves input where not stamped
    ones_full = np.ones((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    inits.append(make_const("ones_full", ones_full))
    nodes.append(helper.make_node("Sub", ["ones_full", "any_stamp"], ["not_stamp"], name="not_stamp"))
    nodes.append(helper.make_node("Mul", [INPUT_NAME, "not_stamp"], ["passthrough"], name="passthrough"))

    # 7. output = stamp_sum + passthrough
    nodes.append(helper.make_node("Add", ["stamp_sum", "passthrough"], ["combined"], name="combine"))

    # 8. Scale + bias for threshold firing
    two_c = make_const("two_c", np.array([2.0], dtype=np.float32))
    neg_half = make_const("neg_half", np.array([-0.5], dtype=np.float32))
    inits.extend([two_c, neg_half])
    nodes.append(helper.make_node("Mul", ["combined", "two_c"], ["scaled"], name="sc"))
    nodes.append(helper.make_node("Add", ["scaled", "neg_half"], [OUTPUT_NAME], name="bias"))

    return make_model(nodes, inits, doc="task 075: template stamping at 1-markers")


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return build()


__all__ = ["build", "generate"]
