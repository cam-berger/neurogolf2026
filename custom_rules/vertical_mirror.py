"""Vertical mirror generators for 10-row grids.

Top-half mirror: output[:5] = input[:5], output[5:] = flipud(input[:5])
Bottom-half mirror: output[5:] = input[5:], output[:5] = flipud(input[5:])
"""

from __future__ import annotations

import onnx
from onnx import helper

from generators.base import INPUT_NAME, OUTPUT_NAME, make_int_const, make_model
from pipeline.loader import HEIGHT


def build_vertical_mirror(which: str) -> onnx.ModelProto:
    if which == "top":
        idx = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0] + list(range(10, HEIGHT))
    elif which == "bottom":
        idx = [9, 8, 7, 6, 5, 5, 6, 7, 8, 9] + list(range(10, HEIGHT))
    else:
        raise ValueError(which)
    idx_const = make_int_const("row_idx", idx)
    nodes = [helper.make_node("Gather", [INPUT_NAME, "row_idx"], [OUTPUT_NAME],
                              axis=2, name="gather_rows")]
    return make_model(nodes, [idx_const], doc=f"vertical mirror ({which})")


__all__ = ["build_vertical_mirror"]
