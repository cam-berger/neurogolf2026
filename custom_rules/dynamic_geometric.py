"""Dynamic-shape geometric generators for variable-size grids.

Opset 10 Equal only supports int types. We use |R - target| < 0.5 instead.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from generators.base import INPUT_NAME, OUTPUT_NAME, make_const, make_int_const, make_model
from pipeline.loader import HEIGHT, WIDTH, CHANNELS


def _width_detect_nodes(axis_name: str, reduce_axes: list[int]):
    """Build nodes that detect rightmost active index along an axis.

    Returns (nodes, inits). Produces tensor "{axis_name}_R" with the max active index.
    """
    nodes = []
    inits = []
    nodes.append(helper.make_node(
        "ReduceMax", [INPUT_NAME], [f"{axis_name}_max"],
        axes=reduce_axes, keepdims=0, name=f"{axis_name}_reduce",
    ))
    zero_const = make_const(f"{axis_name}_zero", np.array([0.0], dtype=np.float32))
    inits.append(zero_const)
    nodes.append(helper.make_node(
        "Greater", [f"{axis_name}_max", f"{axis_name}_zero"],
        [f"{axis_name}_active_bool"], name=f"{axis_name}_gt",
    ))
    nodes.append(helper.make_node(
        "Cast", [f"{axis_name}_active_bool"], [f"{axis_name}_active"],
        to=TensorProto.FLOAT, name=f"{axis_name}_cast",
    ))
    range_30 = make_const(f"{axis_name}_range", np.arange(30, dtype=np.float32))
    inits.append(range_30)
    nodes.append(helper.make_node(
        "Mul", [f"{axis_name}_active", f"{axis_name}_range"],
        [f"{axis_name}_weighted"], name=f"{axis_name}_mul",
    ))
    nodes.append(helper.make_node(
        "ReduceMax", [f"{axis_name}_weighted"], [f"{axis_name}_R"],
        axes=[0], keepdims=1, name=f"{axis_name}_Rmax",
    ))
    return nodes, inits


def _float_eq_gate(R_name: str, target_val: float, tag: str):
    """Build |R - target| < 0.5 gate as float [1,1,1,1]. Returns (nodes, inits)."""
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


def _sum_contribs(contrib_names, nodes):
    cur = contrib_names[0]
    for i, name in enumerate(contrib_names[1:], 1):
        out = OUTPUT_NAME if i == len(contrib_names) - 1 else f"sum_{i}"
        nodes.append(helper.make_node("Add", [cur, name], [out], name=f"add_{i}"))
        cur = out


def _build_flip_h(widths: list[int]) -> onnx.ModelProto:
    nodes, inits = _width_detect_nodes("col", [0, 1, 2])


    contrib_names = []
    for w in widths:
        tag = f"w{w}"
        gn, gi = _float_eq_gate("col_R", float(w - 1), tag)
        nodes += gn; inits += gi

        P = np.zeros((WIDTH, WIDTH), dtype=np.float32)
        for i in range(w):
            P[i, w - 1 - i] = 1.0
        p_const = make_const(f"P_{tag}", P)
        inits.append(p_const)
        nodes.append(helper.make_node("MatMul", [INPUT_NAME, f"P_{tag}"], [f"flip_{tag}"], name=f"mm_{tag}"))
        nodes.append(helper.make_node("Mul", [f"gate_{tag}", f"flip_{tag}"], [f"c_{tag}"], name=f"mul_{tag}"))
        contrib_names.append(f"c_{tag}")

    _sum_contribs(contrib_names, nodes)
    return make_model(nodes, inits, doc=f"dynamic flip_h widths={widths}")


def _build_flip_v(heights: list[int]) -> onnx.ModelProto:
    nodes, inits = _width_detect_nodes("row", [0, 1, 3])


    contrib_names = []
    for h in heights:
        tag = f"h{h}"
        gn, gi = _float_eq_gate("row_R", float(h - 1), tag)
        nodes += gn; inits += gi

        idx = list(range(h - 1, -1, -1)) + list(range(h, HEIGHT))
        idx_const = make_int_const(f"idx_{tag}", idx)
        inits.append(idx_const)
        nodes.append(helper.make_node("Gather", [INPUT_NAME, f"idx_{tag}"], [f"flip_{tag}"], axis=2, name=f"gather_{tag}"))
        mask = np.zeros((1, 1, HEIGHT, 1), dtype=np.float32)
        mask[0, 0, :h, 0] = 1.0
        mask_const = make_const(f"mask_{tag}", mask)
        inits.append(mask_const)
        nodes.append(helper.make_node("Mul", [f"flip_{tag}", f"mask_{tag}"], [f"fmask_{tag}"], name=f"mmask_{tag}"))
        nodes.append(helper.make_node("Mul", [f"gate_{tag}", f"fmask_{tag}"], [f"c_{tag}"], name=f"mul_{tag}"))
        contrib_names.append(f"c_{tag}")

    _sum_contribs(contrib_names, nodes)
    return make_model(nodes, inits, doc=f"dynamic flip_v heights={heights}")


def _build_transpose(sizes: list[int]) -> onnx.ModelProto:
    nodes, inits = _width_detect_nodes("col", [0, 1, 2])

    nodes.append(helper.make_node("Transpose", [INPUT_NAME], ["t_full"], perm=[0, 1, 3, 2], name="transpose_hw"))

    contrib_names = []
    for s in sizes:
        tag = f"s{s}"
        gn, gi = _float_eq_gate("col_R", float(s - 1), tag)
        nodes += gn; inits += gi
        mask = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
        mask[0, 0, :s, :s] = 1.0
        mask_const = make_const(f"mask_{tag}", mask)
        inits.append(mask_const)
        nodes.append(helper.make_node("Mul", ["t_full", f"mask_{tag}"], [f"tmask_{tag}"], name=f"mmask_{tag}"))
        nodes.append(helper.make_node("Mul", [f"gate_{tag}", f"tmask_{tag}"], [f"c_{tag}"], name=f"mul_{tag}"))
        contrib_names.append(f"c_{tag}")

    _sum_contribs(contrib_names, nodes)
    return make_model(nodes, inits, doc=f"dynamic transpose sizes={sizes}")


def _build_scale_up(configs: list[tuple[int, int, int, int]]) -> onnx.ModelProto:
    nodes, inits = _width_detect_nodes("col", [0, 1, 2])


    contrib_names = []
    for hi, wi, ho, wo in configs:
        tag = f"{hi}x{wi}"
        gn, gi = _float_eq_gate("col_R", float(wi - 1), tag)
        nodes += gn; inits += gi

        row_idx = [min(r // 2, hi - 1) if r < ho else 0 for r in range(HEIGHT)]
        col_idx = [min(c // 2, wi - 1) if c < wo else 0 for c in range(WIDTH)]
        ri_c = make_int_const(f"ri_{tag}", row_idx)
        ci_c = make_int_const(f"ci_{tag}", col_idx)
        inits.extend([ri_c, ci_c])
        nodes.append(helper.make_node("Gather", [INPUT_NAME, f"ri_{tag}"], [f"rup_{tag}"], axis=2, name=f"rg_{tag}"))
        nodes.append(helper.make_node("Gather", [f"rup_{tag}", f"ci_{tag}"], [f"sc_{tag}"], axis=3, name=f"cg_{tag}"))
        mask = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
        mask[0, 0, :ho, :wo] = 1.0
        mask_const = make_const(f"mask_{tag}", mask)
        inits.append(mask_const)
        nodes.append(helper.make_node("Mul", [f"sc_{tag}", f"mask_{tag}"], [f"smask_{tag}"], name=f"mmask_{tag}"))
        nodes.append(helper.make_node("Mul", [f"gate_{tag}", f"smask_{tag}"], [f"c_{tag}"], name=f"mul_{tag}"))
        contrib_names.append(f"c_{tag}")

    _sum_contribs(contrib_names, nodes)
    return make_model(nodes, inits, doc=f"dynamic scale_up 2x configs={configs}")


def build_for_task(task_id: int, task: dict) -> onnx.ModelProto | None:
    from pipeline.loader import get_all_pairs
    pairs = get_all_pairs(task)
    shapes = set()
    for inp, out in pairs:
        shapes.add((len(inp), len(inp[0]), len(out), len(out[0])))

    if task_id == 150:
        return _build_flip_h(sorted(set(s[1] for s in shapes)))
    elif task_id == 155:
        return _build_flip_v(sorted(set(s[0] for s in shapes)))
    elif task_id == 241:
        return _build_transpose(sorted(set(s[0] for s in shapes)))
    elif task_id == 307:
        return _build_scale_up(sorted(shapes))
    return None


__all__ = ["build_for_task"]
