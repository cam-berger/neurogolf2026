"""Task 040: wall-orientation detection + nearest-wall recolor.

Fixed 10x10 grid. Walls form either:
- vertical walls (col 0 and col 9 are colored), or
- horizontal walls (row 0 and row 9 are colored).

Interior cells (between the walls) with a non-wall color get recolored to
whichever wall is closer.

Orientation detection: is_vert = input[0,9] == input[9,9] (both same wall color).

Rule:
- For vert: val2 = input[0,0] (left wall), val1 = input[0,9] (right wall).
  Interior cell at (r, c) with c >= 5 → val1, else → val2.
- For horz: val2 = input[0,0] (top wall), val1 = input[9,0] (bottom wall).
  Interior cell at (r, c) with r >= 5 → val1, else → val2.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from generators.base import INPUT_NAME, OUTPUT_NAME, make_const, make_int_const, make_model
from pipeline.loader import HEIGHT, WIDTH, CHANNELS


GRID = 10


def _slice_cell(row: int, col: int, tag: str, inits: list, nodes: list):
    """Slice input at a single cell → [1, 10, 1, 1]."""
    starts = make_int_const(f"s_{tag}", [0, 0, row, col])
    ends = make_int_const(f"e_{tag}", [1, CHANNELS, row + 1, col + 1])
    axes = make_int_const(f"a_{tag}", [0, 1, 2, 3])
    steps = make_int_const(f"st_{tag}", [1, 1, 1, 1])
    inits.extend([starts, ends, axes, steps])
    nodes.append(helper.make_node("Slice",
        [INPUT_NAME, f"s_{tag}", f"e_{tag}", f"a_{tag}", f"st_{tag}"],
        [f"cell_{tag}"], name=f"slice_{tag}"))
    return f"cell_{tag}"


def build() -> onnx.ModelProto:
    nodes = []
    inits = []

    # Extract corner cells as [1, 10, 1, 1] vectors
    c_0_0 = _slice_cell(0, 0, "00", inits, nodes)      # val2 (always)
    c_0_9 = _slice_cell(0, GRID - 1, "09", inits, nodes)  # val1 if vert
    c_9_0 = _slice_cell(GRID - 1, 0, "90", inits, nodes)  # val1 if horz
    c_9_9 = _slice_cell(GRID - 1, GRID - 1, "99", inits, nodes)  # right-bot

    # is_vert: dot(c_0_9, c_9_9) > 0.5. Compute via Mul + ReduceSum.
    nodes.append(helper.make_node("Mul", [c_0_9, c_9_9], ["dot_mul"], name="dot_mul"))
    nodes.append(helper.make_node("ReduceSum", ["dot_mul"], ["dot_scalar"],
                                   axes=[0, 1, 2, 3], keepdims=1, name="dot_sum"))
    # is_vert_bool = dot_scalar > 0.5
    half_c = make_const("half_c", np.array([0.5], dtype=np.float32))
    inits.append(half_c)
    nodes.append(helper.make_node("Greater", ["dot_scalar", "half_c"], ["is_vert_bool"], name="is_vert_gt"))
    nodes.append(helper.make_node("Cast", ["is_vert_bool"], ["is_vert"],
                                   to=TensorProto.FLOAT, name="is_vert_cast"))

    # Compute "1 - is_vert" for horz weighting
    one_c = make_const("one_c", np.array([1.0], dtype=np.float32))
    inits.append(one_c)
    nodes.append(helper.make_node("Sub", ["one_c", "is_vert"], ["is_horz"], name="is_horz_sub"))

    # Compute nz_mask = ReduceMax over channels 1..9 (binary [1, 1, 30, 30])
    s_ch = make_int_const("s_ch", [1])
    e_ch = make_int_const("e_ch", [CHANNELS])
    a_ch = make_int_const("a_ch", [1])
    st_ch = make_int_const("st_ch", [1])
    inits.extend([s_ch, e_ch, a_ch, st_ch])
    nodes.append(helper.make_node("Slice", [INPUT_NAME, "s_ch", "e_ch", "a_ch", "st_ch"],
                                   ["nz_ch"], name="slice_nz"))
    nodes.append(helper.make_node("ReduceMax", ["nz_ch"], ["nz_mask_pre"],
                                   axes=[1], keepdims=1, name="nz_max"))
    nodes.append(helper.make_node("Clip", ["nz_mask_pre"], ["nz_mask"],
                                   min=0.0, max=1.0, name="clip_nz"))

    # not_nz_mask = 1 - nz_mask
    ones_full = np.ones((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    inits.append(make_const("ones_full", ones_full))
    nodes.append(helper.make_node("Sub", ["ones_full", "nz_mask"], ["not_nz_mask"], name="not_nz_sub"))

    # Precomputed region masks (all [1, 1, 30, 30])
    # vert walls: col 0 or col 9 at rows 0..9
    vert_wall = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    vert_wall[0, 0, :GRID, 0] = 1.0
    vert_wall[0, 0, :GRID, GRID - 1] = 1.0
    inits.append(make_const("vert_wall_mask", vert_wall))

    # vert interior left: cols 1..4 at rows 0..9
    vert_left = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    vert_left[0, 0, :GRID, 1:5] = 1.0
    inits.append(make_const("vert_left_mask", vert_left))

    # vert interior right: cols 5..8 at rows 0..9
    vert_right = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    vert_right[0, 0, :GRID, 5:GRID - 1] = 1.0
    inits.append(make_const("vert_right_mask", vert_right))

    # horz walls: row 0 or row 9 at cols 0..9
    horz_wall = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    horz_wall[0, 0, 0, :GRID] = 1.0
    horz_wall[0, 0, GRID - 1, :GRID] = 1.0
    inits.append(make_const("horz_wall_mask", horz_wall))

    horz_top = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    horz_top[0, 0, 1:5, :GRID] = 1.0
    inits.append(make_const("horz_top_mask", horz_top))

    horz_bot = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    horz_bot[0, 0, 5:GRID - 1, :GRID] = 1.0
    inits.append(make_const("horz_bot_mask", horz_bot))

    # grid_mask: 1 at cells in 10x10 area (r in 0..9, c in 0..9)
    grid_m = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    grid_m[0, 0, :GRID, :GRID] = 1.0
    inits.append(make_const("grid_mask", grid_m))

    # Build vert_out:
    # 1. walls: input * vert_wall_mask
    nodes.append(helper.make_node("Mul", [INPUT_NAME, "vert_wall_mask"], ["vert_walls"], name="vert_walls"))
    # 2. left interior nz: left_mask * nz_mask → cells to color val2
    nodes.append(helper.make_node("Mul", ["vert_left_mask", "nz_mask"], ["vert_left_nz"], name="vert_left_nz"))
    # multiply by val2 vector (c_0_0) to get channels
    nodes.append(helper.make_node("Mul", ["vert_left_nz", c_0_0], ["vert_left_contrib"], name="vl_contrib"))
    # 3. right interior nz → val1 (c_0_9)
    nodes.append(helper.make_node("Mul", ["vert_right_mask", "nz_mask"], ["vert_right_nz"], name="vert_right_nz"))
    nodes.append(helper.make_node("Mul", ["vert_right_nz", c_0_9], ["vert_right_contrib"], name="vr_contrib"))
    # 4. empty grid cells: input * not_nz_mask (preserves channel 0 at grid-empty cells, zero at padding)
    nodes.append(helper.make_node("Mul", [INPUT_NAME, "not_nz_mask"], ["empty_copy"], name="empty_copy"))
    # Combine vert_out
    nodes.append(helper.make_node("Add", ["vert_walls", "vert_left_contrib"], ["v_s1"], name="v_s1"))
    nodes.append(helper.make_node("Add", ["v_s1", "vert_right_contrib"], ["v_s2"], name="v_s2"))
    nodes.append(helper.make_node("Add", ["v_s2", "empty_copy"], ["vert_out"], name="vert_out_add"))

    # Build horz_out:
    nodes.append(helper.make_node("Mul", [INPUT_NAME, "horz_wall_mask"], ["horz_walls"], name="horz_walls"))
    nodes.append(helper.make_node("Mul", ["horz_top_mask", "nz_mask"], ["horz_top_nz"], name="horz_top_nz"))
    nodes.append(helper.make_node("Mul", ["horz_top_nz", c_0_0], ["horz_top_contrib"], name="ht_contrib"))
    nodes.append(helper.make_node("Mul", ["horz_bot_mask", "nz_mask"], ["horz_bot_nz"], name="horz_bot_nz"))
    nodes.append(helper.make_node("Mul", ["horz_bot_nz", c_9_0], ["horz_bot_contrib"], name="hb_contrib"))
    nodes.append(helper.make_node("Add", ["horz_walls", "horz_top_contrib"], ["h_s1"], name="h_s1"))
    nodes.append(helper.make_node("Add", ["h_s1", "horz_bot_contrib"], ["h_s2"], name="h_s2"))
    nodes.append(helper.make_node("Add", ["h_s2", "empty_copy"], ["horz_out"], name="horz_out_add"))

    # Select: final = is_vert * vert_out + is_horz * horz_out
    nodes.append(helper.make_node("Mul", ["is_vert", "vert_out"], ["v_weighted"], name="v_weight"))
    nodes.append(helper.make_node("Mul", ["is_horz", "horz_out"], ["h_weighted"], name="h_weight"))
    nodes.append(helper.make_node("Add", ["v_weighted", "h_weighted"], ["combined"], name="combine"))

    # Scale + bias to pass threshold
    two_c = make_const("two_c", np.array([2.0], dtype=np.float32))
    neg_half = make_const("neg_half", np.array([-0.5], dtype=np.float32))
    inits.extend([two_c, neg_half])
    nodes.append(helper.make_node("Mul", ["combined", "two_c"], ["scaled"], name="sc"))
    nodes.append(helper.make_node("Add", ["scaled", "neg_half"], [OUTPUT_NAME], name="bias"))

    return make_model(nodes, inits, doc="task 040: wall orientation + nearest-wall recolor")


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return build()


__all__ = ["build", "generate"]
