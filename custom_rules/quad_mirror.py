"""Quad mirror generator: output is 2x2 grid of [input, fliplr, flipud, rot180]."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from generators.base import INPUT_NAME, OUTPUT_NAME, make_const, make_int_const, make_model
from pipeline.loader import HEIGHT, WIDTH


def build_quad_mirror(h: int, w: int) -> onnx.ModelProto:
    """Output: vstack(hstack(input, fliplr(input)), hstack(flipud(input), rot180(input)))."""
    col_idx = list(range(w)) + list(range(w - 1, -1, -1)) + [0] * (WIDTH - 2 * w)
    row_idx = list(range(h)) + list(range(h - 1, -1, -1)) + [0] * (HEIGHT - 2 * h)

    col_c = make_int_const("col_idx", col_idx)
    row_c = make_int_const("row_idx", row_idx)

    nodes = [
        helper.make_node("Gather", [INPUT_NAME, "col_idx"], ["col_mirror"],
                          axis=3, name="col_gather"),
        helper.make_node("Gather", ["col_mirror", "row_idx"], ["quad"],
                          axis=2, name="row_gather"),
    ]
    mask = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    mask[:, :, :2 * h, :2 * w] = 1.0
    mask_c = make_const("mask", mask)
    nodes.append(helper.make_node("Mul", ["quad", "mask"], [OUTPUT_NAME], name="mask_mul"))

    return make_model(nodes, [col_c, row_c, mask_c], doc=f"quad_mirror {h}x{w}")


__all__ = ["build_quad_mirror"]
