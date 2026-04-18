"""Geometric generators: rot90 / rot180 / rot270 / flip_h / flip_v / transpose.

The network extracts the grid region via Slice, applies a transform (Gather
with a reversed-index constant, optionally preceded by Transpose), and pads
back to [1, 10, 30, 30]. All indices and pad widths are baked in as
constants, which requires the task's input and output shapes to be fixed
across every (train / test / arc-gen) pair.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import onnx
from onnx import helper

from generators.base import (
    INPUT_NAME,
    OUTPUT_NAME,
    NetworkGenerator,
    make_int_const,
    make_model,
    task_shape_signature,
)
from pipeline.loader import HEIGHT, WIDTH

SubFamily = Literal["rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose"]


def _slice_to_grid(input_name: str, h: int, w: int, prefix: str):
    """Emit nodes + inits that slice [1,10,30,30] -> [1,10,h,w] (top-left region)."""
    starts = make_int_const(f"{prefix}_slice_starts", [0, 0])
    ends = make_int_const(f"{prefix}_slice_ends", [h, w])
    axes = make_int_const(f"{prefix}_slice_axes", [2, 3])
    steps = make_int_const(f"{prefix}_slice_steps", [1, 1])
    node = helper.make_node(
        "Slice",
        [input_name, starts.name, ends.name, axes.name, steps.name],
        [f"{prefix}_sliced"],
        name=f"{prefix}_slice",
    )
    return [node], [starts, ends, axes, steps], f"{prefix}_sliced"


def _reverse_axis(input_name: str, axis: int, length: int, prefix: str):
    idx = make_int_const(f"{prefix}_rev_idx", list(range(length - 1, -1, -1)))
    node = helper.make_node(
        "Gather",
        [input_name, idx.name],
        [f"{prefix}_reversed"],
        name=f"{prefix}_gather",
        axis=axis,
    )
    return [node], [idx], f"{prefix}_reversed"


def _transpose_hw(input_name: str, prefix: str):
    node = helper.make_node(
        "Transpose",
        [input_name],
        [f"{prefix}_transposed"],
        name=f"{prefix}_transpose",
        perm=[0, 1, 3, 2],
    )
    return [node], [], f"{prefix}_transposed"


def _pad_to_grid(input_name: str, h_in: int, w_in: int, output_name: str, prefix: str):
    """Opset-10 Pad with `pads` as attribute. Pads to [1,10,30,30] at bottom/right."""
    pads_attr = [0, 0, 0, 0, 0, 0, HEIGHT - h_in, WIDTH - w_in]
    node = helper.make_node(
        "Pad",
        [input_name],
        [output_name],
        name=f"{prefix}_pad",
        mode="constant",
        pads=pads_attr,
        value=0.0,
    )
    return [node]


def _apply_transform(sub: SubFamily, inp: str, h: int, w: int, prefix: str):
    """Apply the geometric transform to a [1,10,h,w] tensor.

    Returns (nodes, inits, out_name, (h_out, w_out)).
    """
    nodes: list = []
    inits: list = []
    cur = inp

    if sub == "flip_h":
        n, i, cur = _reverse_axis(cur, axis=3, length=w, prefix=prefix)
        nodes += n; inits += i
        return nodes, inits, cur, (h, w)

    if sub == "flip_v":
        n, i, cur = _reverse_axis(cur, axis=2, length=h, prefix=prefix)
        nodes += n; inits += i
        return nodes, inits, cur, (h, w)

    if sub == "rot180":
        n, i, cur = _reverse_axis(cur, axis=2, length=h, prefix=f"{prefix}_v")
        nodes += n; inits += i
        n, i, cur = _reverse_axis(cur, axis=3, length=w, prefix=f"{prefix}_h")
        nodes += n; inits += i
        return nodes, inits, cur, (h, w)

    if sub == "transpose":
        n, i, cur = _transpose_hw(cur, prefix)
        nodes += n; inits += i
        return nodes, inits, cur, (w, h)

    if sub == "rot90":  # clockwise: fliplr(transpose)
        n, i, cur = _transpose_hw(cur, prefix)
        nodes += n; inits += i
        # After transpose, shape is (w, h); reverse axis 3 (length = h)
        n, i, cur = _reverse_axis(cur, axis=3, length=h, prefix=f"{prefix}_rev")
        nodes += n; inits += i
        return nodes, inits, cur, (w, h)

    if sub == "rot270":  # 90 CCW: flipud(transpose)
        n, i, cur = _transpose_hw(cur, prefix)
        nodes += n; inits += i
        # After transpose, shape is (w, h); reverse axis 2 (length = w)
        n, i, cur = _reverse_axis(cur, axis=2, length=w, prefix=f"{prefix}_rev")
        nodes += n; inits += i
        return nodes, inits, cur, (w, h)

    raise ValueError(f"unknown sub-family {sub}")


class GeometricGenerator(NetworkGenerator):
    """Single-transform geometric generator. Instantiate once per sub-family."""

    def __init__(self, sub_family: SubFamily):
        self.sub_family = sub_family
        self.family = sub_family

    def can_generate(self, task: dict, features: dict) -> bool:
        flag_key = f"is_{self.sub_family}"
        if not features.get(flag_key):
            return False
        return task_shape_signature(task) is not None

    def generate(self, task: dict, features: dict) -> onnx.ModelProto | None:
        shape = task_shape_signature(task)
        if shape is None:
            return None
        (h_in, w_in), (h_out, w_out) = shape

        nodes: list = []
        inits: list = []

        slice_nodes, slice_inits, cur = _slice_to_grid(INPUT_NAME, h_in, w_in, prefix="crop")
        nodes += slice_nodes; inits += slice_inits

        tf_nodes, tf_inits, cur, (ho, wo) = _apply_transform(
            self.sub_family, cur, h_in, w_in, prefix=self.sub_family
        )
        nodes += tf_nodes; inits += tf_inits

        if (ho, wo) != (h_out, w_out):
            return None  # feature says the transform matches; shape says otherwise

        nodes += _pad_to_grid(cur, ho, wo, OUTPUT_NAME, prefix=self.sub_family)

        return make_model(nodes, inits, doc=f"geometric: {self.sub_family} for {h_in}x{w_in} -> {h_out}x{w_out}")


__all__ = ["GeometricGenerator"]
