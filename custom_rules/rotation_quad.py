"""Rotation quad generator: output = [[id, rot270], [rot90, rot180]] of input (square)."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from generators.base import INPUT_NAME, OUTPUT_NAME, make_const, make_int_const, make_model
from pipeline.loader import HEIGHT, WIDTH


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


def _build_one_size(n: int, tag: str):
    """Build 4-path quad for a specific size n. Returns (nodes, inits, out_name)."""
    nodes = []
    inits = []

    # Transpose input (shared - could be moved outside if we do all sizes)
    # But we use a unique tag per size, so name it uniquely
    nodes.append(helper.make_node("Transpose", [INPUT_NAME], [f"T_{tag}"],
                                   perm=[0, 1, 3, 2], name=f"transpose_{tag}"))

    # TL: identity, masked
    mask_tl = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    mask_tl[0, 0, :n, :n] = 1.0
    inits.append(make_const(f"mtl_{tag}", mask_tl))
    nodes.append(helper.make_node("Mul", [INPUT_NAME, f"mtl_{tag}"], [f"tl_{tag}"], name=f"mul_tl_{tag}"))

    # TR: rot270, output[r, c] = T[r, 2n-1-c] for r<n, n<=c<2n
    row_idx_tr = [r if r < n else 0 for r in range(HEIGHT)]
    col_idx_tr = [2*n-1-c if n <= c < 2*n else 0 for c in range(WIDTH)]
    inits.append(make_int_const(f"tri_row_{tag}", row_idx_tr))
    inits.append(make_int_const(f"tri_col_{tag}", col_idx_tr))
    nodes.append(helper.make_node("Gather", [f"T_{tag}", f"tri_row_{tag}"], [f"tr_rev_{tag}"],
                                   axis=2, name=f"tr_gr_{tag}"))
    nodes.append(helper.make_node("Gather", [f"tr_rev_{tag}", f"tri_col_{tag}"], [f"tr_raw_{tag}"],
                                   axis=3, name=f"tr_gc_{tag}"))
    mask_tr = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    mask_tr[0, 0, :n, n:2*n] = 1.0
    inits.append(make_const(f"mtr_{tag}", mask_tr))
    nodes.append(helper.make_node("Mul", [f"tr_raw_{tag}", f"mtr_{tag}"], [f"tr_{tag}"], name=f"mul_tr_{tag}"))

    # BL: rot90, output[r, c] = T[2n-1-r, c] for n<=r<2n, c<n
    row_idx_bl = [2*n-1-r if n <= r < 2*n else 0 for r in range(HEIGHT)]
    col_idx_bl = [c if c < n else 0 for c in range(WIDTH)]
    inits.append(make_int_const(f"bli_row_{tag}", row_idx_bl))
    inits.append(make_int_const(f"bli_col_{tag}", col_idx_bl))
    nodes.append(helper.make_node("Gather", [f"T_{tag}", f"bli_row_{tag}"], [f"bl_row_{tag}"],
                                   axis=2, name=f"bl_gr_{tag}"))
    nodes.append(helper.make_node("Gather", [f"bl_row_{tag}", f"bli_col_{tag}"], [f"bl_raw_{tag}"],
                                   axis=3, name=f"bl_gc_{tag}"))
    mask_bl = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    mask_bl[0, 0, n:2*n, :n] = 1.0
    inits.append(make_const(f"mbl_{tag}", mask_bl))
    nodes.append(helper.make_node("Mul", [f"bl_raw_{tag}", f"mbl_{tag}"], [f"bl_{tag}"], name=f"mul_bl_{tag}"))

    # BR: rot180, output[r, c] = input[2n-1-r, 2n-1-c] for n<=r<2n, n<=c<2n
    row_idx_br = [2*n-1-r if n <= r < 2*n else 0 for r in range(HEIGHT)]
    col_idx_br = [2*n-1-c if n <= c < 2*n else 0 for c in range(WIDTH)]
    inits.append(make_int_const(f"bri_row_{tag}", row_idx_br))
    inits.append(make_int_const(f"bri_col_{tag}", col_idx_br))
    nodes.append(helper.make_node("Gather", [INPUT_NAME, f"bri_row_{tag}"], [f"br_row_{tag}"],
                                   axis=2, name=f"br_gr_{tag}"))
    nodes.append(helper.make_node("Gather", [f"br_row_{tag}", f"bri_col_{tag}"], [f"br_raw_{tag}"],
                                   axis=3, name=f"br_gc_{tag}"))
    mask_br = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    mask_br[0, 0, n:2*n, n:2*n] = 1.0
    inits.append(make_const(f"mbr_{tag}", mask_br))
    nodes.append(helper.make_node("Mul", [f"br_raw_{tag}", f"mbr_{tag}"], [f"br_{tag}"], name=f"mul_br_{tag}"))

    # Sum
    nodes.append(helper.make_node("Add", [f"tl_{tag}", f"tr_{tag}"], [f"s1_{tag}"], name=f"s1_{tag}"))
    nodes.append(helper.make_node("Add", [f"s1_{tag}", f"bl_{tag}"], [f"s2_{tag}"], name=f"s2_{tag}"))
    nodes.append(helper.make_node("Add", [f"s2_{tag}", f"br_{tag}"], [f"quad_{tag}"], name=f"sfinal_{tag}"))

    return nodes, inits, f"quad_{tag}"


def build_rotation_quad(sizes: list[int]) -> onnx.ModelProto:
    """Dynamic-size rotation quad for square inputs, supporting multiple sizes."""
    nodes, inits = _col_detector()

    contrib_names = []
    for n in sizes:
        tag = f"n{n}"
        sub_nodes, sub_inits, out_name = _build_one_size(n, tag)
        nodes += sub_nodes
        inits += sub_inits

        gn, gi = _float_eq_gate("col_R", float(n - 1), tag)
        nodes += gn
        inits += gi

        nodes.append(helper.make_node("Mul", [f"gate_{tag}", out_name], [f"contrib_{tag}"],
                                       name=f"g_mul_{tag}"))
        contrib_names.append(f"contrib_{tag}")

    # Sum all contributions
    cur = contrib_names[0]
    for i, name in enumerate(contrib_names[1:], 1):
        out = OUTPUT_NAME if i == len(contrib_names) - 1 else f"gsum_{i}"
        nodes.append(helper.make_node("Add", [cur, name], [out], name=f"gadd_{i}"))
        cur = out
    if len(contrib_names) == 1:
        nodes.append(helper.make_node("Identity", [contrib_names[0]], [OUTPUT_NAME], name="copy"))

    return make_model(nodes, inits, doc=f"dynamic rotation_quad sizes={sizes}")


__all__ = ["build_rotation_quad"]
