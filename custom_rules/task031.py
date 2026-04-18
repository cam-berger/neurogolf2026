"""Task 031: output = input cropped to bounding box of non-zero cells.

Implementation:
1. Compute row_mask [30]: 1 if row r has any non-zero channel
2. Compute col_mask [30]: 1 if col c has any non-zero channel
3. r_min = ReduceMin over (r where row_mask[r]=1) — use mask * r + (1-mask)*LARGE, then ReduceMin
4. r_max = ReduceMax over (r where row_mask[r]=1) — use mask * r + (1-mask)*(-1), then ReduceMax
5. Same for c_min, c_max
6. Gather rows [r_min, r_min+1, ..., r_min+29] clipped to [0, 29]
7. Gather cols [c_min, c_min+1, ..., c_min+29] clipped to [0, 29]
8. Mask rows >= (r_max - r_min + 1) and cols >= (c_max - c_min + 1) to zero
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

    # 1. Compute any NON-ZERO-COLOR activity per cell: slice channels 1..9, ReduceMax.
    # Channel 0 (color 0 = "background") is not counted as "non-zero".
    starts_ch = make_int_const("s_ch", [1])
    ends_ch = make_int_const("e_ch", [CHANNELS])
    axes_ch = make_int_const("a_ch", [1])
    steps_ch = make_int_const("st_ch", [1])
    inits.extend([starts_ch, ends_ch, axes_ch, steps_ch])
    nodes.append(helper.make_node("Slice", [INPUT_NAME, "s_ch", "e_ch", "a_ch", "st_ch"],
                                   ["nz_channels"], name="slice_nz"))
    nodes.append(helper.make_node("ReduceMax", ["nz_channels"], ["any_ch"],
                                   axes=[1], keepdims=1, name="any_ch"))
    nodes.append(helper.make_node("Clip", ["any_ch"], ["nz_mask"], min=0.0, max=1.0, name="clip_nz"))

    # 2. row_mask [1, 1, 30, 1]: any col has activity
    nodes.append(helper.make_node("ReduceMax", ["nz_mask"], ["row_mask"],
                                   axes=[3], keepdims=1, name="row_max"))
    # col_mask [1, 1, 1, 30]
    nodes.append(helper.make_node("ReduceMax", ["nz_mask"], ["col_mask"],
                                   axes=[2], keepdims=1, name="col_max"))

    # 3. r_min: for each row, score = r * row_mask + LARGE * (1 - row_mask); then ReduceMin
    # range values along row axis [1, 1, 30, 1]
    range_h = np.arange(HEIGHT, dtype=np.float32).reshape(1, 1, HEIGHT, 1)
    inits.append(make_const("range_h", range_h))
    range_w = np.arange(WIDTH, dtype=np.float32).reshape(1, 1, 1, WIDTH)
    inits.append(make_const("range_w", range_w))

    # score_r_min = r * row_mask + LARGE * (1 - row_mask)
    # = r * row_mask - LARGE * row_mask + LARGE
    # = (r - LARGE) * row_mask + LARGE
    LARGE = 1000.0
    large_c = make_const("large_c", np.array([LARGE], dtype=np.float32))
    inits.append(large_c)
    # neg_large for broadcasts
    neg_large_c = make_const("neg_large_c", np.array([-LARGE], dtype=np.float32))
    inits.append(neg_large_c)

    # For r_min: score = row_mask * r + (1-row_mask) * LARGE
    #   = row_mask * (r - LARGE) + LARGE
    # We want ReduceMin over axis 2 (height axis).
    nodes.append(helper.make_node("Add", ["range_h", "neg_large_c"], ["range_h_minus_L"], name="rhML"))
    nodes.append(helper.make_node("Mul", ["row_mask", "range_h_minus_L"], ["rscore_min_a"], name="rsca"))
    nodes.append(helper.make_node("Add", ["rscore_min_a", "large_c"], ["rscore_min"], name="rsmin"))
    nodes.append(helper.make_node("ReduceMin", ["rscore_min"], ["r_min_f"],
                                   axes=[2], keepdims=1, name="red_rmin"))

    # For r_max: score = row_mask * r + (1-row_mask) * (-1)
    #   = row_mask * (r + 1) - 1
    neg_one_c = make_const("neg_one_c", np.array([-1.0], dtype=np.float32))
    one_c = make_const("one_c", np.array([1.0], dtype=np.float32))
    inits.extend([neg_one_c, one_c])
    nodes.append(helper.make_node("Add", ["range_h", "one_c"], ["range_h_plus1"], name="rhp1"))
    nodes.append(helper.make_node("Mul", ["row_mask", "range_h_plus1"], ["rsmax_a"], name="rsmax_a"))
    nodes.append(helper.make_node("Add", ["rsmax_a", "neg_one_c"], ["rscore_max"], name="rscmax"))
    nodes.append(helper.make_node("ReduceMax", ["rscore_max"], ["r_max_f"],
                                   axes=[2], keepdims=1, name="red_rmax"))

    # For c_min, c_max: analogous on axis 3
    nodes.append(helper.make_node("Add", ["range_w", "neg_large_c"], ["range_w_minus_L"], name="rwML"))
    nodes.append(helper.make_node("Mul", ["col_mask", "range_w_minus_L"], ["cscore_min_a"], name="csca"))
    nodes.append(helper.make_node("Add", ["cscore_min_a", "large_c"], ["cscore_min"], name="csmin"))
    nodes.append(helper.make_node("ReduceMin", ["cscore_min"], ["c_min_f"],
                                   axes=[3], keepdims=1, name="red_cmin"))

    nodes.append(helper.make_node("Add", ["range_w", "one_c"], ["range_w_plus1"], name="rwp1"))
    nodes.append(helper.make_node("Mul", ["col_mask", "range_w_plus1"], ["csmax_a"], name="csmax_a"))
    nodes.append(helper.make_node("Add", ["csmax_a", "neg_one_c"], ["cscore_max"], name="cscmax"))
    nodes.append(helper.make_node("ReduceMax", ["cscore_max"], ["c_max_f"],
                                   axes=[3], keepdims=1, name="red_cmax"))

    # Now r_min_f, r_max_f are [1, 1, 1, 1] scalars. Similarly c_min_f, c_max_f.

    # 6. Build row gather indices: [r_min, r_min+1, ..., r_min+29], clipped to [0, 29]
    # r_min_f broadcast to shape [30]: need Reshape to [1] then Add with range_30 [30] → [30]
    # But we want int64 for Gather.

    # First reshape scalars to [1]
    shape_1 = make_int_const("shape_1", [1])
    inits.append(shape_1)
    nodes.append(helper.make_node("Reshape", ["r_min_f", "shape_1"], ["r_min_f1"], name="rsh_rmin"))
    nodes.append(helper.make_node("Reshape", ["r_max_f", "shape_1"], ["r_max_f1"], name="rsh_rmax"))
    nodes.append(helper.make_node("Reshape", ["c_min_f", "shape_1"], ["c_min_f1"], name="rsh_cmin"))
    nodes.append(helper.make_node("Reshape", ["c_max_f", "shape_1"], ["c_max_f1"], name="rsh_cmax"))

    # range as 1D float
    range_1d_h = make_const("range_1d_h", np.arange(HEIGHT, dtype=np.float32))
    range_1d_w = make_const("range_1d_w", np.arange(WIDTH, dtype=np.float32))
    inits.extend([range_1d_h, range_1d_w])

    # row_idx_f = range_1d_h + r_min (broadcast)
    nodes.append(helper.make_node("Add", ["range_1d_h", "r_min_f1"], ["row_idx_raw"], name="rir"))
    # Clip to [0, 29]
    max_idx_h = make_const("max_idx_h", np.array([HEIGHT - 1], dtype=np.float32))
    inits.append(max_idx_h)
    # clip via Min
    nodes.append(helper.make_node("Min", ["row_idx_raw", "max_idx_h"], ["row_idx_clipped"], name="ric"))
    # Cast to int64
    nodes.append(helper.make_node("Cast", ["row_idx_clipped"], ["row_idx"],
                                   to=TensorProto.INT64, name="cast_ri"))

    # Same for col
    nodes.append(helper.make_node("Add", ["range_1d_w", "c_min_f1"], ["col_idx_raw"], name="cir"))
    max_idx_w = make_const("max_idx_w", np.array([WIDTH - 1], dtype=np.float32))
    inits.append(max_idx_w)
    nodes.append(helper.make_node("Min", ["col_idx_raw", "max_idx_w"], ["col_idx_clipped"], name="cic"))
    nodes.append(helper.make_node("Cast", ["col_idx_clipped"], ["col_idx"],
                                   to=TensorProto.INT64, name="cast_ci"))

    # 7. Gather rows then cols
    nodes.append(helper.make_node("Gather", [INPUT_NAME, "row_idx"], ["rgath"], axis=2, name="gath_r"))
    nodes.append(helper.make_node("Gather", ["rgath", "col_idx"], ["cgath"], axis=3, name="gath_c"))

    # 8. Mask: height = r_max - r_min + 1, width = c_max - c_min + 1
    # mask_row[r] = 1 if r < height, else 0
    # height broadcast from [1] scalar
    nodes.append(helper.make_node("Sub", ["r_max_f1", "r_min_f1"], ["rdiff"], name="rd"))
    nodes.append(helper.make_node("Add", ["rdiff", "one_c"], ["height_scalar"], name="hs"))
    nodes.append(helper.make_node("Sub", ["c_max_f1", "c_min_f1"], ["cdiff"], name="cd"))
    nodes.append(helper.make_node("Add", ["cdiff", "one_c"], ["width_scalar"], name="ws"))

    # row_mask_out[r] = Less(range_1d_h, height_scalar) → bool → Cast to float
    nodes.append(helper.make_node("Less", ["range_1d_h", "height_scalar"], ["rm_bool"], name="rm_less"))
    nodes.append(helper.make_node("Cast", ["rm_bool"], ["rm_f"], to=TensorProto.FLOAT, name="rm_cast"))
    nodes.append(helper.make_node("Less", ["range_1d_w", "width_scalar"], ["cm_bool"], name="cm_less"))
    nodes.append(helper.make_node("Cast", ["cm_bool"], ["cm_f"], to=TensorProto.FLOAT, name="cm_cast"))

    # Reshape masks for broadcast
    rm_shape = make_int_const("rm_shape", [1, 1, HEIGHT, 1])
    cm_shape = make_int_const("cm_shape", [1, 1, 1, WIDTH])
    inits.extend([rm_shape, cm_shape])
    nodes.append(helper.make_node("Reshape", ["rm_f", "rm_shape"], ["rm_4d"], name="rm_rsh"))
    nodes.append(helper.make_node("Reshape", ["cm_f", "cm_shape"], ["cm_4d"], name="cm_rsh"))

    # Apply masks
    nodes.append(helper.make_node("Mul", ["cgath", "rm_4d"], ["masked_r"], name="mr"))
    nodes.append(helper.make_node("Mul", ["masked_r", "cm_4d"], [OUTPUT_NAME], name="mc"))

    return make_model(nodes, inits, doc="task 031: crop to bbox of non-zero cells")


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return build()


__all__ = ["build", "generate"]
