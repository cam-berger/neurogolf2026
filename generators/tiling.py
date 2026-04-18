"""Tiling generator: integer nearest-neighbor upscale.

Emits Slice -> Resize(nearest) -> Pad. Needs the task to have fixed input
and output shapes across every pair so the slice/pad constants are valid.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from generators.base import (
    INPUT_NAME,
    OUTPUT_NAME,
    NetworkGenerator,
    make_int_const,
    make_model,
    task_shape_signature,
)
from pipeline.loader import HEIGHT, WIDTH


def _scales_initializer(name: str, scales: list[float]) -> onnx.TensorProto:
    arr = np.asarray(scales, dtype=np.float32)
    return helper.make_tensor(name=name, data_type=TensorProto.FLOAT, dims=[len(scales)], vals=arr.flatten().tolist())


class TilingGenerator(NetworkGenerator):
    family = "scale_up"

    def can_generate(self, task: dict, features: dict) -> bool:
        if not features.get("output_is_input_scaled"):
            return False
        if features.get("scale_factor_h", 1.0) < 2 or features.get("scale_factor_w", 1.0) < 2:
            return False
        return task_shape_signature(task) is not None

    def generate(self, task: dict, features: dict) -> onnx.ModelProto | None:
        shape = task_shape_signature(task)
        if shape is None:
            return None
        (h_in, w_in), (h_out, w_out) = shape
        sh, sw = int(features["scale_factor_h"]), int(features["scale_factor_w"])
        if h_in * sh != h_out or w_in * sw != w_out:
            return None
        if h_out > HEIGHT or w_out > WIDTH:
            return None

        # 1. Slice to grid region [1,10,h_in,w_in]
        starts = make_int_const("tile_slice_starts", [0, 0])
        ends = make_int_const("tile_slice_ends", [h_in, w_in])
        axes = make_int_const("tile_slice_axes", [2, 3])
        steps = make_int_const("tile_slice_steps", [1, 1])
        slice_node = helper.make_node(
            "Slice",
            [INPUT_NAME, starts.name, ends.name, axes.name, steps.name],
            ["tile_sliced"],
            name="tile_slice",
        )

        # 2. Resize nearest by [1, 1, sh, sw]
        scales = _scales_initializer("tile_scales", [1.0, 1.0, float(sh), float(sw)])
        resize_node = helper.make_node(
            "Resize",
            ["tile_sliced", scales.name],
            ["tile_resized"],
            name="tile_resize",
            mode="nearest",
        )

        # 3. Pad bottom/right back to [1,10,30,30]
        pads_attr = [0, 0, 0, 0, 0, 0, HEIGHT - h_out, WIDTH - w_out]
        pad_node = helper.make_node(
            "Pad",
            ["tile_resized"],
            [OUTPUT_NAME],
            name="tile_pad",
            mode="constant",
            pads=pads_attr,
            value=0.0,
        )

        return make_model(
            [slice_node, resize_node, pad_node],
            [starts, ends, axes, steps, scales],
            doc=f"scale up {h_in}x{w_in} -> {h_out}x{w_out} by ({sh},{sw})",
        )


__all__ = ["TilingGenerator"]
