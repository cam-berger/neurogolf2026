"""Color-remap generator: 1x1 Conv with a 10x10 permutation matrix."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from generators.base import INPUT_NAME, OUTPUT_NAME, NetworkGenerator, make_const, make_model
from pipeline.loader import CHANNELS


def _permutation_weight(mapping: dict[int, int]) -> np.ndarray:
    """Build a [out=10, in=10, 1, 1] conv weight such that channel `in` maps to channel `mapping[in]`."""
    w = np.zeros((CHANNELS, CHANNELS, 1, 1), dtype=np.float32)
    for in_color, out_color in mapping.items():
        if 0 <= in_color < CHANNELS and 0 <= out_color < CHANNELS:
            w[out_color, in_color, 0, 0] = 1.0
    return w


class ColorRemapGenerator(NetworkGenerator):
    family = "color_remap"

    def can_generate(self, task: dict, features: dict) -> bool:
        return bool(features.get("is_color_permutation")) and features.get("color_permutation_map") is not None

    def generate(self, task: dict, features: dict) -> onnx.ModelProto | None:
        mapping = features.get("color_permutation_map")
        if not mapping:
            return None
        weight = _permutation_weight(mapping)
        w_init = make_const("W", weight)
        conv = helper.make_node(
            "Conv",
            [INPUT_NAME, "W"],
            [OUTPUT_NAME],
            name="color_remap",
            kernel_shape=[1, 1],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
        )
        return make_model([conv], [w_init], doc=f"color remap: {mapping}")


__all__ = ["ColorRemapGenerator"]
