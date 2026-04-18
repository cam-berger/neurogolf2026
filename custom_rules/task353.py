"""Task 353: single 3 and single 4 in grid. The 3 moves one step toward the 4.
Output: 3 at new position, 4 unchanged, everything else 0.

Implementation:
1. Extract mask_3 (channel 3) and mask_4 (channel 4), each [1, 1, 30, 30].
2. Compute row_mask_3 [1, 1, 30, 1] and row_mask_4, similar for col.
3. Compute gates for each direction (dr, dc) in {-1, 0, 1}^2:
   gate_dr_dc = (r4 relative position to r3 == dr) AND (c4 relative position to c3 == dc)
4. For each direction, shift mask_3 by (dr, dc) via Gather.
5. Sum all gated shifts → new position of 3.
6. Output: input + (new_3_mask - old_3_mask) with channel 3 adjustments and channel 0 adjustments.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from generators.base import INPUT_NAME, OUTPUT_NAME, make_const, make_int_const, make_model
from pipeline.loader import HEIGHT, WIDTH, CHANNELS


def _slice_channel(node_input: str, ch: int, tag: str):
    """Extract a single channel → [1, 1, 30, 30]."""
    nodes = []
    inits = []
    starts = make_int_const(f"s_{tag}", [ch])
    ends = make_int_const(f"e_{tag}", [ch + 1])
    axes = make_int_const(f"a_{tag}", [1])
    steps = make_int_const(f"st_{tag}", [1])
    inits.extend([starts, ends, axes, steps])
    nodes.append(helper.make_node("Slice",
        [node_input, f"s_{tag}", f"e_{tag}", f"a_{tag}", f"st_{tag}"],
        [f"{tag}_mask"], name=f"slice_{tag}"))
    return nodes, inits, f"{tag}_mask"


def _shift_2d(mask_name: str, dr: int, dc: int, tag: str):
    """Shift [1, 1, 30, 30] mask by (dr, dc). Returns (nodes, inits, output_name).
    new[r, c] = mask[r - dr, c - dc] (with 0 padding for out-of-bounds).
    """
    nodes = []
    inits = []
    # Row shift via Gather: new[r] = old[r - dr]
    # indices: for each output row r, the input row is r - dr if valid, else 0.
    # But we want 0-padding for invalid, so use index 0 and multiply by mask later.
    row_idx = []
    valid_row = []
    for r in range(HEIGHT):
        src = r - dr
        if 0 <= src < HEIGHT:
            row_idx.append(src)
            valid_row.append(1.0)
        else:
            row_idx.append(0)
            valid_row.append(0.0)
    row_idx_c = make_int_const(f"rowidx_{tag}", row_idx)
    row_mask = np.array(valid_row, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    row_mask_c = make_const(f"rowmask_{tag}", row_mask)
    inits.extend([row_idx_c, row_mask_c])
    nodes.append(helper.make_node("Gather", [mask_name, f"rowidx_{tag}"], [f"rshift_{tag}"],
                                   axis=2, name=f"rshift_{tag}"))
    nodes.append(helper.make_node("Mul", [f"rshift_{tag}", f"rowmask_{tag}"], [f"rshift_m_{tag}"],
                                   name=f"rshift_m_{tag}"))

    # Col shift
    col_idx = []
    valid_col = []
    for c in range(WIDTH):
        src = c - dc
        if 0 <= src < WIDTH:
            col_idx.append(src)
            valid_col.append(1.0)
        else:
            col_idx.append(0)
            valid_col.append(0.0)
    col_idx_c = make_int_const(f"colidx_{tag}", col_idx)
    col_mask = np.array(valid_col, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    col_mask_c = make_const(f"colmask_{tag}", col_mask)
    inits.extend([col_idx_c, col_mask_c])
    nodes.append(helper.make_node("Gather", [f"rshift_m_{tag}", f"colidx_{tag}"], [f"cshift_{tag}"],
                                   axis=3, name=f"cshift_{tag}"))
    nodes.append(helper.make_node("Mul", [f"cshift_{tag}", f"colmask_{tag}"], [f"shifted_{tag}"],
                                   name=f"cshift_m_{tag}"))
    return nodes, inits, f"shifted_{tag}"


def build() -> onnx.ModelProto:
    nodes = []
    inits = []

    # Step 1: Extract mask_3 and mask_4
    sn3, si3, m3 = _slice_channel(INPUT_NAME, 3, "ch3")
    sn4, si4, m4 = _slice_channel(INPUT_NAME, 4, "ch4")
    nodes += sn3 + sn4
    inits += si3 + si4

    # Step 2: Row and col masks
    # row_mask_3[r] = any c with m3[r, c] = 1 → ReduceMax axis=[3], shape [1, 1, 30, 1]
    nodes.append(helper.make_node("ReduceMax", [m3], ["rm3"], axes=[3], keepdims=1, name="rm3"))
    nodes.append(helper.make_node("ReduceMax", [m3], ["cm3"], axes=[2], keepdims=1, name="cm3"))
    nodes.append(helper.make_node("ReduceMax", [m4], ["rm4"], axes=[3], keepdims=1, name="rm4"))
    nodes.append(helper.make_node("ReduceMax", [m4], ["cm4"], axes=[2], keepdims=1, name="cm4"))

    # Step 3: Compute direction gates
    # For row: compute cumul_below_4[r] = sum_{j > r} rm4[j], cumul_above_4[r] = sum_{j < r} rm4[j]
    # These are [1, 1, 30, 1].
    # Cumul below: L_below[i, j] = 1 if j > i (strict upper). cumul_below = L_below @ rm4.
    L_below = np.zeros((HEIGHT, HEIGHT), dtype=np.float32)
    for i in range(HEIGHT):
        for j in range(i + 1, HEIGHT):
            L_below[i, j] = 1.0
    L_above = L_below.T.copy()
    inits.append(make_const("L_below", L_below))
    inits.append(make_const("L_above", L_above))
    # MatMul needs [..., 30, 30] @ [1, 1, 30, 1] → [..., 30, 1]
    # L is [30, 30], rm4 is [1, 1, 30, 1]. MatMul broadcasts.
    nodes.append(helper.make_node("MatMul", ["L_below", "rm4"], ["cumul_below_4"], name="mm_cb4"))
    nodes.append(helper.make_node("MatMul", ["L_above", "rm4"], ["cumul_above_4"], name="mm_ca4"))

    # gate_dr1 = sum(rm3 * cumul_below_4) — scalar: 1 if r4 > r3
    nodes.append(helper.make_node("Mul", ["rm3", "cumul_below_4"], ["gdr1_prod"], name="mul_gdr1"))
    nodes.append(helper.make_node("ReduceSum", ["gdr1_prod"], ["gate_dr1"],
                                   axes=[2, 3], keepdims=1, name="red_gdr1"))
    nodes.append(helper.make_node("Mul", ["rm3", "cumul_above_4"], ["gdr_1_prod"], name="mul_gdrn1"))
    nodes.append(helper.make_node("ReduceSum", ["gdr_1_prod"], ["gate_dr_1"],
                                   axes=[2, 3], keepdims=1, name="red_gdr_1"))
    # gate_dr0 = sum(rm3 * rm4)
    nodes.append(helper.make_node("Mul", ["rm3", "rm4"], ["gdr0_prod"], name="mul_gdr0"))
    nodes.append(helper.make_node("ReduceSum", ["gdr0_prod"], ["gate_dr0"],
                                   axes=[2, 3], keepdims=1, name="red_gdr0"))

    # Column gates: need cumul_right_4 and cumul_left_4 along axis 3
    # cm3, cm4 are [1, 1, 1, 30].
    # MatMul: cm_row @ L_right where L_right[i, j] = 1 if j > i (upper tri, no diag)
    # (cm @ L)[j] = sum_i cm[i] * L[i, j]
    # We want: cumul_right[c] = sum_{j > c} cm4[j]. Not quite — need cm shape to align.
    # cm4 is [1, 1, 1, 30]. L_right [30, 30]: (cm4 @ L_right)[c] = sum_i cm4[i] * L_right[i, c]
    #   = sum_i cm4[i] if c > i = sum_{i < c} cm4[i] = cumul_left[c]
    # So cm4 @ L_right = cumul_left
    # And cm4 @ L_right^T where L_right^T[i, j] = 1 if i > j:
    #   = sum_i cm4[i] if i > c = sum_{i > c} cm4[i] = cumul_right[c]
    # So use L_right and L_left properly.
    # L_upper[i, j] = 1 if j > i (upper strict). cm4 @ L_upper gives cumul_left[c] = sum_{i < c} cm4[i].
    # L_lower[i, j] = 1 if j < i (lower strict). cm4 @ L_lower gives cumul_right[c] = sum_{i > c} cm4[i].
    # Actually let me re-derive:
    # (cm4 @ L_upper)[..., c] = sum over i of cm4[..., i] * L_upper[i, c]
    # L_upper[i, c] = 1 if c > i = 1 if i < c → sum_{i < c} cm4[i]
    # That's cumul_left. To get cumul_right[c] = sum_{i > c}, we need L[i, c] = 1 if i > c → lower triangular
    L_lower = np.tril(np.ones((WIDTH, WIDTH), dtype=np.float32), k=-1)  # L_lower[i, j] = 1 if i > j
    inits.append(make_const("L_lower", L_lower))
    L_upper = np.triu(np.ones((WIDTH, WIDTH), dtype=np.float32), k=1)
    inits.append(make_const("L_upper", L_upper))

    nodes.append(helper.make_node("MatMul", ["cm4", "L_lower"], ["cumul_right_4"], name="mm_cr4"))
    nodes.append(helper.make_node("MatMul", ["cm4", "L_upper"], ["cumul_left_4"], name="mm_cl4"))

    # gate_dc1 = sum(cm3 * cumul_right_4)
    nodes.append(helper.make_node("Mul", ["cm3", "cumul_right_4"], ["gdc1_prod"], name="mul_gdc1"))
    nodes.append(helper.make_node("ReduceSum", ["gdc1_prod"], ["gate_dc1"],
                                   axes=[2, 3], keepdims=1, name="red_gdc1"))
    nodes.append(helper.make_node("Mul", ["cm3", "cumul_left_4"], ["gdc_1_prod"], name="mul_gdcn1"))
    nodes.append(helper.make_node("ReduceSum", ["gdc_1_prod"], ["gate_dc_1"],
                                   axes=[2, 3], keepdims=1, name="red_gdc_1"))
    nodes.append(helper.make_node("Mul", ["cm3", "cm4"], ["gdc0_prod"], name="mul_gdc0"))
    nodes.append(helper.make_node("ReduceSum", ["gdc0_prod"], ["gate_dc0"],
                                   axes=[2, 3], keepdims=1, name="red_gdc0"))

    # Step 4: For each direction, shift mask_3 and weight by gate
    gate_names = {(-1, -1): ("gate_dr_1", "gate_dc_1"), (-1, 0): ("gate_dr_1", "gate_dc0"),
                  (-1, 1): ("gate_dr_1", "gate_dc1"), (0, -1): ("gate_dr0", "gate_dc_1"),
                  (0, 0): ("gate_dr0", "gate_dc0"), (0, 1): ("gate_dr0", "gate_dc1"),
                  (1, -1): ("gate_dr1", "gate_dc_1"), (1, 0): ("gate_dr1", "gate_dc0"),
                  (1, 1): ("gate_dr1", "gate_dc1")}

    contrib_names = []
    for (dr, dc), (gdr, gdc) in gate_names.items():
        tag = f"d{dr}_{dc}".replace("-", "m")
        # shift mask_3 by (dr, dc)
        shift_nodes, shift_inits, shifted = _shift_2d(m3, dr, dc, tag)
        nodes += shift_nodes
        inits += shift_inits
        # gate = gdr * gdc
        nodes.append(helper.make_node("Mul", [gdr, gdc], [f"g_{tag}"], name=f"gmul_{tag}"))
        # contribution = gate * shifted
        nodes.append(helper.make_node("Mul", [f"g_{tag}", shifted], [f"c_{tag}"], name=f"cmul_{tag}"))
        contrib_names.append(f"c_{tag}")

    # Sum all contributions to get new_3_mask [1, 1, 30, 30]
    cur = contrib_names[0]
    for i, n in enumerate(contrib_names[1:], 1):
        out = f"new3_{i}" if i < len(contrib_names) - 1 else "new3_mask"
        nodes.append(helper.make_node("Add", [cur, n], [out], name=f"add_{i}"))
        cur = out

    # Step 5: Build output.
    # We want:
    # - channel 3 fired at new_3_mask positions (not at original 3 position)
    # - channel 4 fired at original 4 position (unchanged)
    # - channel 0 fired everywhere else on grid, including the now-vacated original 3 position
    # - Everything outside grid: no channels fired
    #
    # Strategy:
    # - Start with input, remove the original 3 (zero out channel 3 at original 3 pos, but put channel 0 there)
    # - Add channel 3 at new_3_mask position, subtract channel 0 there

    # Build identity base with bias
    id_w = np.zeros((CHANNELS, CHANNELS, 1, 1), dtype=np.float32)
    for c in range(CHANNELS):
        id_w[c, c, 0, 0] = 2.0
    inits.append(make_const("id_w", id_w))
    id_b = np.full((CHANNELS,), -0.5, dtype=np.float32)
    inits.append(make_const("id_b", id_b))
    nodes.append(helper.make_node("Conv", [INPUT_NAME, "id_w", "id_b"], ["id_out"],
                                   kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], name="id_conv"))

    # Remove channel 3 at original 3 position, add channel 0 there
    # At position of m3: channel 3 -= 2, channel 0 += 2
    remove_3_w = np.zeros((CHANNELS, 1, 1, 1), dtype=np.float32)
    remove_3_w[3, 0, 0, 0] = -4.0  # cancel +2 (was +2 - 0.5 = 1.5, now = -2.5)
    remove_3_w[0, 0, 0, 0] = +2.0  # add +2 to channel 0 (was 0 - 0.5 = -0.5, now = +1.5)
    inits.append(make_const("remove_3_w", remove_3_w))
    nodes.append(helper.make_node("Conv", [m3, "remove_3_w"], ["rm3_overlay"],
                                   kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], name="rm3_conv"))

    # Add channel 3 at new_3_mask position, remove channel 0 there
    add_3_w = np.zeros((CHANNELS, 1, 1, 1), dtype=np.float32)
    add_3_w[3, 0, 0, 0] = +4.0  # channel 3: was -0.5 → +3.5
    add_3_w[0, 0, 0, 0] = -4.0  # channel 0: was +1.5 → -2.5
    inits.append(make_const("add_3_w", add_3_w))
    nodes.append(helper.make_node("Conv", ["new3_mask", "add_3_w"], ["add3_overlay"],
                                   kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], name="add3_conv"))

    # Combine: id_out + rm3_overlay + add3_overlay
    nodes.append(helper.make_node("Add", ["id_out", "rm3_overlay"], ["s1"], name="add_s1"))
    nodes.append(helper.make_node("Add", ["s1", "add3_overlay"], [OUTPUT_NAME], name="add_final"))

    return make_model(nodes, inits, doc="task 353: 3 moves toward 4")


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return build()


__all__ = ["build", "generate"]
