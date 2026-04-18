"""Reflection+concat generators: output = concat(input, flip(input)) on some axis."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from generators.base import INPUT_NAME, OUTPUT_NAME, make_const, make_int_const, make_model
from pipeline.loader import HEIGHT, WIDTH, CHANNELS


def _col_detector():
    nodes = []
    inits = []
    nodes.append(helper.make_node("ReduceMax", [INPUT_NAME], ["col_max"],
                                   axes=[0, 1, 2], keepdims=0, name="col_reduce"))
    zero_c = make_const("col_zero", np.array([0.0], dtype=np.float32))
    inits.append(zero_c)
    nodes.append(helper.make_node("Greater", ["col_max", "col_zero"], ["col_bool"], name="col_gt"))
    nodes.append(helper.make_node("Cast", ["col_bool"], ["col_active"],
                                   to=TensorProto.FLOAT, name="col_cast"))
    range_c = make_const("col_range", np.arange(30, dtype=np.float32))
    inits.append(range_c)
    nodes.append(helper.make_node("Mul", ["col_active", "col_range"], ["col_w"], name="col_mul"))
    nodes.append(helper.make_node("ReduceMax", ["col_w"], ["col_R"],
                                   axes=[0], keepdims=1, name="col_Rmax"))
    return nodes, inits


def _row_detector():
    nodes = []
    inits = []
    nodes.append(helper.make_node("ReduceMax", [INPUT_NAME], ["row_max"],
                                   axes=[0, 1, 3], keepdims=0, name="row_reduce"))
    zero_c = make_const("row_zero", np.array([0.0], dtype=np.float32))
    inits.append(zero_c)
    nodes.append(helper.make_node("Greater", ["row_max", "row_zero"], ["row_bool"], name="row_gt"))
    nodes.append(helper.make_node("Cast", ["row_bool"], ["row_active"],
                                   to=TensorProto.FLOAT, name="row_cast"))
    range_r = make_const("row_range", np.arange(30, dtype=np.float32))
    inits.append(range_r)
    nodes.append(helper.make_node("Mul", ["row_active", "row_range"], ["row_w"], name="row_mul"))
    nodes.append(helper.make_node("ReduceMax", ["row_w"], ["row_R"],
                                   axes=[0], keepdims=1, name="row_Rmax"))
    return nodes, inits


def _float_eq_gate(R_name: str, target: float, tag: str):
    nodes = []
    inits = []
    t_c = make_const(f"t_{tag}", np.array([target], dtype=np.float32))
    h_c = make_const(f"h_{tag}", np.array([0.5], dtype=np.float32))
    inits.extend([t_c, h_c])
    nodes.append(helper.make_node("Sub", [R_name, f"t_{tag}"], [f"d_{tag}"], name=f"sub_{tag}"))
    nodes.append(helper.make_node("Abs", [f"d_{tag}"], [f"ad_{tag}"], name=f"abs_{tag}"))
    nodes.append(helper.make_node("Less", [f"ad_{tag}", f"h_{tag}"], [f"e_{tag}"], name=f"less_{tag}"))
    nodes.append(helper.make_node("Cast", [f"e_{tag}"], [f"ef_{tag}"],
                                   to=TensorProto.FLOAT, name=f"cast_{tag}"))
    shape_c = make_int_const(f"sh_{tag}", [1, 1, 1, 1])
    inits.append(shape_c)
    nodes.append(helper.make_node("Reshape", [f"ef_{tag}", f"sh_{tag}"], [f"gate_{tag}"],
                                   name=f"rsh_{tag}"))
    return nodes, inits


def _sum_contribs(names, nodes):
    cur = names[0]
    for i, n in enumerate(names[1:], 1):
        out = OUTPUT_NAME if i == len(names) - 1 else f"sum_{i}"
        nodes.append(helper.make_node("Add", [cur, n], [out], name=f"add_{i}"))
        cur = out
    if len(names) == 1:
        nodes.append(helper.make_node("Identity", [names[0]], [OUTPUT_NAME], name="copy"))


def build_hflip_right(widths: list[int]) -> onnx.ModelProto:
    """output = np.hstack([input, np.fliplr(input)])

    For a grid of width w, output column c < w is input[c], output column c in [w, 2w)
    is input[w-1-(c-w)] = input[2w-1-c]. Output columns >= 2w are zero.
    """
    nodes, inits = _col_detector()
    contrib_names = []
    for w in widths:
        tag = f"w{w}"
        gn, gi = _float_eq_gate("col_R", float(w - 1), tag)
        nodes += gn; inits += gi

        idx = []
        for c in range(WIDTH):
            if c < w:
                idx.append(c)
            elif c < 2 * w:
                idx.append(2 * w - 1 - c)
            else:
                idx.append(0)  # doesn't matter, we'll mask
        idx_c = make_int_const(f"idx_{tag}", idx)
        inits.append(idx_c)
        nodes.append(helper.make_node("Gather", [INPUT_NAME, f"idx_{tag}"], [f"g_{tag}"],
                                       axis=3, name=f"gather_{tag}"))
        mask = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
        mask[:, :, :, :2 * w] = 1.0
        mask_c = make_const(f"mask_{tag}", mask)
        inits.append(mask_c)
        nodes.append(helper.make_node("Mul", [f"g_{tag}", f"mask_{tag}"], [f"m_{tag}"],
                                       name=f"mul_{tag}"))
        nodes.append(helper.make_node("Mul", [f"gate_{tag}", f"m_{tag}"], [f"c_{tag}"],
                                       name=f"gmul_{tag}"))
        contrib_names.append(f"c_{tag}")

    _sum_contribs(contrib_names, nodes)
    return make_model(nodes, inits, doc=f"hflip_right widths={widths}")


def build_vflip_down(heights: list[int]) -> onnx.ModelProto:
    """output = np.vstack([input, np.flipud(input)])"""
    nodes, inits = _row_detector()
    contrib_names = []
    for h in heights:
        tag = f"h{h}"
        gn, gi = _float_eq_gate("row_R", float(h - 1), tag)
        nodes += gn; inits += gi

        idx = []
        for r in range(HEIGHT):
            if r < h:
                idx.append(r)
            elif r < 2 * h:
                idx.append(2 * h - 1 - r)
            else:
                idx.append(0)
        idx_c = make_int_const(f"idx_{tag}", idx)
        inits.append(idx_c)
        nodes.append(helper.make_node("Gather", [INPUT_NAME, f"idx_{tag}"], [f"g_{tag}"],
                                       axis=2, name=f"gather_{tag}"))
        mask = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
        mask[:, :, :2 * h, :] = 1.0
        mask_c = make_const(f"mask_{tag}", mask)
        inits.append(mask_c)
        nodes.append(helper.make_node("Mul", [f"g_{tag}", f"mask_{tag}"], [f"m_{tag}"],
                                       name=f"mul_{tag}"))
        nodes.append(helper.make_node("Mul", [f"gate_{tag}", f"m_{tag}"], [f"c_{tag}"],
                                       name=f"gmul_{tag}"))
        contrib_names.append(f"c_{tag}")

    _sum_contribs(contrib_names, nodes)
    return make_model(nodes, inits, doc=f"vflip_down heights={heights}")


def build_vflip_up(heights: list[int]) -> onnx.ModelProto:
    """output = np.vstack([np.flipud(input), input])"""
    nodes, inits = _row_detector()
    contrib_names = []
    for h in heights:
        tag = f"h{h}"
        gn, gi = _float_eq_gate("row_R", float(h - 1), tag)
        nodes += gn; inits += gi

        idx = []
        for r in range(HEIGHT):
            if r < h:
                idx.append(h - 1 - r)  # flipped input at top
            elif r < 2 * h:
                idx.append(r - h)  # original input below
            else:
                idx.append(0)
        idx_c = make_int_const(f"idx_{tag}", idx)
        inits.append(idx_c)
        nodes.append(helper.make_node("Gather", [INPUT_NAME, f"idx_{tag}"], [f"g_{tag}"],
                                       axis=2, name=f"gather_{tag}"))
        mask = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
        mask[:, :, :2 * h, :] = 1.0
        mask_c = make_const(f"mask_{tag}", mask)
        inits.append(mask_c)
        nodes.append(helper.make_node("Mul", [f"g_{tag}", f"mask_{tag}"], [f"m_{tag}"],
                                       name=f"mul_{tag}"))
        nodes.append(helper.make_node("Mul", [f"gate_{tag}", f"m_{tag}"], [f"c_{tag}"],
                                       name=f"gmul_{tag}"))
        contrib_names.append(f"c_{tag}")

    _sum_contribs(contrib_names, nodes)
    return make_model(nodes, inits, doc=f"vflip_up heights={heights}")


__all__ = ["build_hflip_right", "build_vflip_down", "build_vflip_up"]
