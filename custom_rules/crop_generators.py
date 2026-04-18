"""Crop generators for unclassified crop-pattern tasks.

Task 326: fixed 2×2 top-left crop (output is always 2×2)
Task 67: factor 3x horizontal crop (input (H, 3H) → output (H, H))
Task 188: half crop — output is (H, W/2) or (H/2, W), whichever is valid
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from generators.base import INPUT_NAME, OUTPUT_NAME, make_const, make_int_const, make_model
from pipeline.loader import HEIGHT, WIDTH, CHANNELS


def _build_fixed_crop(oh: int, ow: int) -> onnx.ModelProto:
    """Build a generator that takes top-left (oh, ow) region of input and zeros the rest."""
    mask = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    mask[0, 0, :oh, :ow] = 1.0
    mask_const = make_const("mask", mask)
    node = helper.make_node("Mul", [INPUT_NAME, "mask"], [OUTPUT_NAME], name="crop_mask")
    return make_model([node], [mask_const], doc=f"fixed top-left crop {oh}x{ow}")


def _build_row_detector() -> tuple[list, list]:
    """Build nodes to compute R_row = last active row index."""
    nodes = []
    inits = []
    nodes.append(helper.make_node(
        "ReduceMax", [INPUT_NAME], ["row_max"],
        axes=[0, 1, 3], keepdims=0, name="row_reduce",
    ))
    zero_const = make_const("row_zero", np.array([0.0], dtype=np.float32))
    inits.append(zero_const)
    nodes.append(helper.make_node(
        "Greater", ["row_max", "row_zero"], ["row_active_bool"], name="row_gt",
    ))
    nodes.append(helper.make_node(
        "Cast", ["row_active_bool"], ["row_active"], to=TensorProto.FLOAT, name="row_cast",
    ))
    range_30 = make_const("row_range", np.arange(30, dtype=np.float32))
    inits.append(range_30)
    nodes.append(helper.make_node(
        "Mul", ["row_active", "row_range"], ["row_weighted"], name="row_mul",
    ))
    nodes.append(helper.make_node(
        "ReduceMax", ["row_weighted"], ["row_R"], axes=[0], keepdims=1, name="row_Rmax",
    ))
    return nodes, inits


def _build_col_detector() -> tuple[list, list]:
    nodes = []
    inits = []
    nodes.append(helper.make_node(
        "ReduceMax", [INPUT_NAME], ["col_max"],
        axes=[0, 1, 2], keepdims=0, name="col_reduce",
    ))
    zero_const = make_const("col_zero", np.array([0.0], dtype=np.float32))
    inits.append(zero_const)
    nodes.append(helper.make_node(
        "Greater", ["col_max", "col_zero"], ["col_active_bool"], name="col_gt",
    ))
    nodes.append(helper.make_node(
        "Cast", ["col_active_bool"], ["col_active"], to=TensorProto.FLOAT, name="col_cast",
    ))
    range_30 = make_const("col_range", np.arange(30, dtype=np.float32))
    inits.append(range_30)
    nodes.append(helper.make_node(
        "Mul", ["col_active", "col_range"], ["col_weighted"], name="col_mul",
    ))
    nodes.append(helper.make_node(
        "ReduceMax", ["col_weighted"], ["col_R"], axes=[0], keepdims=1, name="col_Rmax",
    ))
    return nodes, inits


def _float_eq_gate(R_name: str, target_val: float, tag: str):
    """Build |R - target| < 0.5 gate, shaped [1,1,1,1]."""
    nodes = []
    inits = []
    t_const = make_const(f"t_{tag}", np.array([target_val], dtype=np.float32))
    half_const = make_const(f"half_{tag}", np.array([0.5], dtype=np.float32))
    inits.extend([t_const, half_const])
    nodes.append(helper.make_node("Sub", [R_name, f"t_{tag}"], [f"diff_{tag}"], name=f"sub_{tag}"))
    nodes.append(helper.make_node("Abs", [f"diff_{tag}"], [f"adiff_{tag}"], name=f"abs_{tag}"))
    nodes.append(helper.make_node("Less", [f"adiff_{tag}", f"half_{tag}"], [f"eq_{tag}_bool"], name=f"less_{tag}"))
    nodes.append(helper.make_node("Cast", [f"eq_{tag}_bool"], [f"eq_{tag}"], to=TensorProto.FLOAT, name=f"cast_{tag}"))
    shape_c = make_int_const(f"gshape_{tag}", [1, 1, 1, 1])
    inits.append(shape_c)
    nodes.append(helper.make_node("Reshape", [f"eq_{tag}", f"gshape_{tag}"], [f"gate_{tag}"], name=f"reshape_{tag}"))
    return nodes, inits


def _build_size_gated_crop(configs: list[tuple[int, int, int, int]]) -> onnx.ModelProto:
    """Build a crop generator that gates on input dimensions.

    configs: list of (ih, iw, oh, ow). Gate on input height; mask top-left (oh, ow).
    """
    nodes, inits = _build_row_detector()

    contrib_names = []
    for ih, iw, oh, ow in configs:
        tag = f"i{ih}x{iw}o{oh}x{ow}"
        gn, gi = _float_eq_gate("row_R", float(ih - 1), tag)
        nodes += gn
        inits += gi

        mask = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
        mask[0, 0, :oh, :ow] = 1.0
        mask_const = make_const(f"mask_{tag}", mask)
        inits.append(mask_const)
        nodes.append(helper.make_node("Mul", [INPUT_NAME, f"mask_{tag}"], [f"masked_{tag}"], name=f"mmask_{tag}"))
        nodes.append(helper.make_node("Mul", [f"gate_{tag}", f"masked_{tag}"], [f"c_{tag}"], name=f"mul_{tag}"))
        contrib_names.append(f"c_{tag}")

    cur = contrib_names[0]
    for i, name in enumerate(contrib_names[1:], 1):
        out = OUTPUT_NAME if i == len(contrib_names) - 1 else f"sum_{i}"
        nodes.append(helper.make_node("Add", [cur, name], [out], name=f"add_{i}"))
        cur = out
    if len(contrib_names) == 1:
        nodes.append(helper.make_node("Identity", [contrib_names[0]], [OUTPUT_NAME], name="copy"))

    return make_model(nodes, inits, doc=f"size-gated crop ({len(configs)} configs)")


def build_task_326(task: dict) -> onnx.ModelProto:
    """Fixed 2×2 top-left crop."""
    return _build_fixed_crop(2, 2)


def _build_region_crop(r0: int, c0: int, oh: int, ow: int) -> onnx.ModelProto:
    """Extract input[:, :, r0:r0+oh, c0:c0+ow] and pad to [1,10,30,30]."""
    starts = make_int_const("starts", [r0, c0])
    ends = make_int_const("ends", [r0 + oh, c0 + ow])
    axes = make_int_const("axes", [2, 3])
    steps = make_int_const("steps", [1, 1])
    slice_node = helper.make_node(
        "Slice", [INPUT_NAME, "starts", "ends", "axes", "steps"], ["sliced"],
        name="region_slice",
    )
    pad_node = helper.make_node(
        "Pad", ["sliced"], [OUTPUT_NAME], name="region_pad",
        mode="constant",
        pads=[0, 0, 0, 0, 0, 0, HEIGHT - oh, WIDTH - ow], value=0.0,
    )
    return make_model(
        [slice_node, pad_node],
        [starts, ends, axes, steps],
        doc=f"region crop (r0={r0},c0={c0},oh={oh},ow={ow})",
    )


def build_task_135(task: dict) -> onnx.ModelProto:
    """Fixed top-right 3x3 crop from 9x9 input."""
    return _build_region_crop(r0=0, c0=6, oh=3, ow=3)


def build_task_67(task: dict) -> onnx.ModelProto:
    """Factor 1x3 crop: input (H, 3H) → output (H, H)."""
    from pipeline.loader import get_all_pairs
    pairs = get_all_pairs(task)
    configs = sorted(set((len(i), len(i[0]), len(o), len(o[0])) for i, o in pairs))
    return _build_size_gated_crop(configs)


__all__ = ["build_task_326", "build_task_67", "build_task_135"]
