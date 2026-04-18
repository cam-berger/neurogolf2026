"""Task 199: single non-zero cell at (r0, c0) with color C.
Output:
  - 4 at every (r, c) with r <= r0 AND (c - c0) is even
  - C at (r0 + 1, c0)

Approach:
  1. non_zero_mask = ReduceMax over channels 1..9 → [1, 1, 30, 30] binary
  2. row_mask[r] = any(non_zero_mask[r, *]) → [1, 1, 30, 1]
  3. col_mask[c] = any(non_zero_mask[*, c]) → [1, 1, 1, 30]
  4. above_mask[r] = cumulative OR from bottom: mark rows 0..r0
     Computed via MatMul with upper-triangular matrix.
  5. parity_mask[c]: cells at cols with same parity as c0
     = OR(even_cols_active, odd_cols_active) where each is detected from col_mask * [even_pattern]
  6. 4_mask = above_mask * parity_mask
  7. shifted_color = Shift input down by 1 row
  8. Combine: output = (input with original color removed) + 4_at_mask + shifted_color
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

    # Step 1: nonzero_mask = sum of channels 1..9 (any color other than "background 0")
    starts = make_int_const("ch_s", [1])
    ends = make_int_const("ch_e", [CHANNELS])
    axes = make_int_const("ch_a", [1])
    steps = make_int_const("ch_st", [1])
    inits.extend([starts, ends, axes, steps])
    nodes.append(helper.make_node("Slice", [INPUT_NAME, "ch_s", "ch_e", "ch_a", "ch_st"],
                                   ["nonzero_ch"], name="slice_ch"))
    nodes.append(helper.make_node("ReduceSum", ["nonzero_ch"], ["nonzero_mask"],
                                   axes=[1], keepdims=1, name="red_ch"))
    # Clip to [0, 1]
    nodes.append(helper.make_node("Clip", ["nonzero_mask"], ["nz"], min=0.0, max=1.0, name="clip_nz"))

    # Step 2: row_mask shape [1, 1, 30, 1]
    nodes.append(helper.make_node("ReduceMax", ["nz"], ["row_mask"],
                                   axes=[3], keepdims=1, name="row_max"))
    # Step 3: col_mask shape [1, 1, 1, 30]
    nodes.append(helper.make_node("ReduceMax", ["nz"], ["col_mask"],
                                   axes=[2], keepdims=1, name="col_max"))

    # Step 4: above_mask[r] = 1 iff any row j >= r has the cell
    # Use MatMul with upper-triangular matrix T [30, 30] where T[i, j] = 1 if j >= i
    T = np.zeros((HEIGHT, HEIGHT), dtype=np.float32)
    for i in range(HEIGHT):
        for j in range(i, HEIGHT):
            T[i, j] = 1.0
    # row_mask: [1, 1, 30, 1], reshape to [1, 1, 30] for matmul
    # Actually, use MatMul: [..., 30, 1] @ [1, 30] is not quite right.
    # Let me use T @ row_mask instead. row_mask shape [1, 1, 30, 1], T shape [30, 30].
    # MatMul(T, row_mask): need broadcasting. Actually easier:
    # Transpose row_mask to [1, 1, 1, 30], then MatMul with T^T of shape [30, 30]:
    #   [1, 1, 1, 30] @ [30, 30] = [1, 1, 1, 30]
    # But we want above_mask[r] = sum_j T[r, j] * row_mask[j].
    # If row_mask is treated as [1, 1, 1, 30] (row vector), then row_mask @ T^T has shape [1, 1, 1, 30]
    # and entry at col r is sum_j T^T[j, r] * row_mask[0, 0, 0, j] = sum_j T[r, j] * row_mask[j]. ✓

    # Reshape row_mask from [1, 1, 30, 1] to [1, 1, 1, 30]
    shape_1130 = make_int_const("shape_1130", [1, 1, 1, HEIGHT])
    inits.append(shape_1130)
    nodes.append(helper.make_node("Reshape", ["row_mask", "shape_1130"], ["row_mask_row"], name="rsh_rowm"))

    T_t = T.T.astype(np.float32)  # transpose so MatMul works correctly
    inits.append(make_const("T_mat", T_t))
    nodes.append(helper.make_node("MatMul", ["row_mask_row", "T_mat"], ["above_mask_row"], name="mm_above"))
    # above_mask_row: [1, 1, 1, 30] where entry at col r = above_mask[r]
    # We want above_mask as [1, 1, 30, 1] (column vector for broadcast with col_mask [1, 1, 1, 30])
    shape_1301 = make_int_const("shape_1301", [1, 1, HEIGHT, 1])
    inits.append(shape_1301)
    nodes.append(helper.make_node("Reshape", ["above_mask_row", "shape_1301"], ["above_mask"], name="rsh_above"))
    # Clip to [0, 1]
    nodes.append(helper.make_node("Clip", ["above_mask"], ["above_mask_b"], min=0.0, max=1.0, name="clip_above"))

    # Step 5: parity_mask. Detect if c0 is even or odd, then pick appropriate column pattern.
    # even_cols_pattern[c] = 1 if c % 2 == 0
    # odd_cols_pattern[c] = 1 if c % 2 == 1
    even_cols = np.zeros((1, 1, 1, WIDTH), dtype=np.float32)
    odd_cols = np.zeros((1, 1, 1, WIDTH), dtype=np.float32)
    for c in range(WIDTH):
        if c % 2 == 0: even_cols[0, 0, 0, c] = 1.0
        else: odd_cols[0, 0, 0, c] = 1.0
    inits.append(make_const("even_cols", even_cols))
    inits.append(make_const("odd_cols", odd_cols))

    # Detect: is c0 at even col? = any(col_mask * even_cols)
    nodes.append(helper.make_node("Mul", ["col_mask", "even_cols"], ["col_mask_even"], name="mul_ce"))
    nodes.append(helper.make_node("ReduceSum", ["col_mask_even"], ["has_even_sum"],
                                   axes=[3], keepdims=1, name="red_ce"))
    nodes.append(helper.make_node("Clip", ["has_even_sum"], ["has_even"], min=0.0, max=1.0, name="clip_he"))
    nodes.append(helper.make_node("Mul", ["col_mask", "odd_cols"], ["col_mask_odd"], name="mul_co"))
    nodes.append(helper.make_node("ReduceSum", ["col_mask_odd"], ["has_odd_sum"],
                                   axes=[3], keepdims=1, name="red_co"))
    nodes.append(helper.make_node("Clip", ["has_odd_sum"], ["has_odd"], min=0.0, max=1.0, name="clip_ho"))

    # parity_mask = has_even * even_cols + has_odd * odd_cols
    nodes.append(helper.make_node("Mul", ["has_even", "even_cols"], ["p_even"], name="mul_pe"))
    nodes.append(helper.make_node("Mul", ["has_odd", "odd_cols"], ["p_odd"], name="mul_po"))
    nodes.append(helper.make_node("Add", ["p_even", "p_odd"], ["parity_mask"], name="add_parity"))

    # Compute grid-extent mask: any channel active anywhere
    nodes.append(helper.make_node("ReduceMax", [INPUT_NAME], ["grid_ext_any"],
                                   axes=[1], keepdims=1, name="grid_max_ch"))
    # grid_ext_any: [1, 1, 30, 30] binary, 1 where any channel is active (= grid area)
    nodes.append(helper.make_node("Clip", ["grid_ext_any"], ["grid_ext"], min=0.0, max=1.0, name="clip_grid"))

    # Step 6: 4_mask = above_mask * parity_mask * grid_ext, shape [1, 1, 30, 30] after broadcast
    nodes.append(helper.make_node("Mul", ["above_mask_b", "parity_mask"], ["fours_pre"], name="mul_fours_pre"))
    nodes.append(helper.make_node("Mul", ["fours_pre", "grid_ext"], ["fours_mask"], name="mul_fours"))

    # Step 7: shifted_color = shift input down by 1 row.
    # Use Gather on axis 2 with indices [0, 0, 1, 2, ..., 28] (first element repeats so original row 0 stays but shifted down).
    # Actually: new[r] = old[r - 1] for r >= 1, and new[0] = 0 (or anything — nothing there).
    # Use Gather with indices [0, 0, 1, 2, ..., 28]. Then zero out new[0] via mask.
    shift_idx = [0] + list(range(HEIGHT - 1))
    inits.append(make_int_const("shift_down_idx", shift_idx))
    nodes.append(helper.make_node("Gather", [INPUT_NAME, "shift_down_idx"], ["shifted"],
                                   axis=2, name="gather_shift"))
    # Zero out row 0 of shifted (the duplicate)
    zero_row0 = np.ones((1, 1, HEIGHT, 1), dtype=np.float32)
    zero_row0[0, 0, 0, 0] = 0.0
    inits.append(make_const("mask_notrow0", zero_row0))
    nodes.append(helper.make_node("Mul", ["shifted", "mask_notrow0"], ["shifted_c"], name="mul_shifted"))

    # Step 8: Remove original color from its position.
    # The shifted output contains everything from original input (minus row 0). We want ONLY the single nonzero cell shifted.
    # Actually since input has only 1 non-zero cell, shifted_c at (r0+1, c0) = input[r0, c0] color C.
    # And shifted_c at (r0, c0) = input[r0-1, c0] = 0 (no cell there).
    # Everywhere else, shifted_c is 0 (since input is 0 everywhere else).
    # Actually wait — shifted_c includes the CHANNEL-0 activations too! In one-hot encoding, every cell
    # has channel 0 active (the "color 0" = background). So shifted_c at every position has channel 0 = 1
    # (shifted from input's channel 0 = 1 everywhere except at the original cell position).

    # Hmm, this is an issue. The scoring semantics:
    # - predicted[cell] = channels fired. 0 channels → "no color" → decoded as 10 (stripped)
    # - We need output to have channel 0 fired for cells that should be color 0
    # But channel 0 fired EVERYWHERE in input (since color 0 is everywhere except at the nonzero cell).

    # Actually yes channel 0 IS active everywhere except at the one colored cell.
    # So in output:
    #   - Cells that should be 0: channel 0 fired
    #   - Cells that should be 4 (in the fours_mask region): channel 4 fired, channel 0 NOT fired
    #   - Cell (r0+1, c0): color C fired, channel 0 NOT fired

    # Let me build the output carefully:
    # 1. Start with input (which has channel 0 everywhere it should, and channel C at (r0, c0))
    # 2. "Remove" the colored cell (zero out all channels at (r0, c0))
    # 3. "Place" fours at fours_mask positions (channel 4 = 1, channel 0 = 0)
    # 4. "Place" shifted color at (r0+1, c0) (channel C = 1 from shifted input)
    #
    # Step 2 is tricky without knowing which channel. Since we want to zero out all channels at the cell,
    # we can multiply input by (1 - nonzero_cell_mask) where nonzero_cell_mask is [1, 1, 30, 30] with 1 only at (r0, c0).
    # Then all channels at that position are zeroed out. Channel 0 will still be zero there too (it was 0 in input).
    #
    # Step 3: in fours_mask positions, set channel 4 = 1 and channel 0 = 0.
    # Step 4: at (r0+1, c0), shifted_c has channel C = 1 and channel 0 = 0. So just ADD shifted_c.

    # First: zero the input at the colored-cell position (keeps channel 0 everywhere else as is)
    # not_cell = 1 - nonzero_mask (it's [1, 1, 30, 30])
    one_tensor = np.ones((1, 1, 1, 1), dtype=np.float32)
    inits.append(make_const("one_scalar", one_tensor))
    nodes.append(helper.make_node("Sub", ["one_scalar", "nz"], ["not_cell"], name="sub_not_cell"))
    # Broadcast not_cell [1, 1, 30, 30] across channels
    nodes.append(helper.make_node("Mul", [INPUT_NAME, "not_cell"], ["input_nocell"], name="mul_nocell"))

    # Step 3: At fours_mask positions, we want channel 4 fired and channel 0 NOT fired.
    # First compute: "has_fours" at each (r, c) = fours_mask = 1 at those cells.
    # We subtract channel 0 at those cells and add channel 4.
    # overlay_4 has channel 4 = +1, channel 0 = -1 at each cell where fours_mask = 1
    overlay_w = np.zeros((CHANNELS, 1, 1, 1), dtype=np.float32)
    overlay_w[4, 0, 0, 0] = 2.0  # push channel 4 above threshold
    overlay_w[0, 0, 0, 0] = -2.0  # pull channel 0 below threshold
    inits.append(make_const("overlay_w", overlay_w))
    nodes.append(helper.make_node("Conv", ["fours_mask", "overlay_w"], ["fours_overlay"],
                                   kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], name="ov_conv"))

    # Add a -0.5 bias to the base path to ensure correct thresholding
    # Build identity path via 1x1 conv with bias -0.5
    id_w = np.zeros((CHANNELS, CHANNELS, 1, 1), dtype=np.float32)
    for c in range(CHANNELS):
        id_w[c, c, 0, 0] = 2.0  # 2 * channel - nothing will threshold correctly
    inits.append(make_const("id_w", id_w))
    id_bias = np.full((CHANNELS,), -0.5, dtype=np.float32)
    inits.append(make_const("id_b", id_bias))
    nodes.append(helper.make_node("Conv", ["input_nocell", "id_w", "id_b"], ["id_base"],
                                   kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], name="id_conv"))

    # Shifted color overlay: shifted_c has channel C active at (r0+1, c0). Multiply by 2 to fire above threshold.
    # Also need to suppress channel 0 at (r0+1, c0).
    # shifted_c has shape [1, 10, 30, 30]. Channel 0 is NOT active at (r0+1, c0) since original cell was color C.
    # So at (r0+1, c0), shifted_c[0]=0, shifted_c[C]=1.
    # After scaling by 2: channel C = 2. Channel 0 = 0.
    # But we need to suppress channel 0 that's currently active at (r0+1, c0) in input_nocell.
    # Use the nonzero_cell_mask shifted down to identify where to suppress channel 0.
    nodes.append(helper.make_node("Gather", ["nz", "shift_down_idx"], ["nz_shift"],
                                   axis=2, name="gather_nz_shift"))
    nodes.append(helper.make_node("Mul", ["nz_shift", "mask_notrow0"], ["nz_shift_c"], name="mul_nz_shift"))
    # Suppress channel 0 at that position
    suppress_w = np.zeros((CHANNELS, 1, 1, 1), dtype=np.float32)
    suppress_w[0, 0, 0, 0] = -2.0
    inits.append(make_const("suppress_w", suppress_w))
    nodes.append(helper.make_node("Conv", ["nz_shift_c", "suppress_w"], ["suppress_0"],
                                   kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], name="sup_conv"))

    # Restrict shifted to just the one shifted cell (mask by nz_shift_c)
    nodes.append(helper.make_node("Mul", ["shifted_c", "nz_shift_c"], ["shifted_restricted"], name="mul_sr"))
    # Scale shifted color
    scale_w = np.zeros((CHANNELS, CHANNELS, 1, 1), dtype=np.float32)
    for c in range(CHANNELS):
        scale_w[c, c, 0, 0] = 2.0
    inits.append(make_const("scale_w", scale_w))
    nodes.append(helper.make_node("Conv", ["shifted_restricted", "scale_w"], ["shifted_scaled"],
                                   kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], name="sc_conv"))

    # Combine everything:
    # output = id_base + fours_overlay + shifted_scaled + suppress_0
    nodes.append(helper.make_node("Add", ["id_base", "fours_overlay"], ["s1"], name="add1"))
    nodes.append(helper.make_node("Add", ["s1", "shifted_scaled"], ["s2"], name="add2"))
    nodes.append(helper.make_node("Add", ["s2", "suppress_0"], [OUTPUT_NAME], name="add3"))

    return make_model(nodes, inits, doc="task 199: single cell gravity with 4-trail")


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return build()


__all__ = ["build", "generate"]
