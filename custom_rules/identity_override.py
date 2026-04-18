"""Identity + Override ONNX compiler for near-identity local rules.

For tasks where most windows map center→same color, this produces a much
smaller model than compile_lut_to_onnx: an identity 1×1 conv copies the
input, while a small set of 5×5 (or kxk) detectors fire on the override
windows and correct via Add.

Architecture:
    input ─┬─ Conv(1×1, 10→10, bias=-0.5) ─ ReLU ───── identity_out ─┐
           │                                                           Add → output
           └─ Conv(k×k, 10→H, bias) ─ ReLU ─ Conv(1×1, H→10) ───────┘

The override router writes +2.0 to the new color and -2.0 to the old center
color, so after Add the identity contribution is cancelled and the new color
takes over.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from generators.base import INPUT_NAME, OUTPUT_NAME, make_const, make_model
from pipeline.loader import CHANNELS, get_all_pairs

from custom_rules.lut import _one_hot_window, extract_lut


def compile_identity_override(lut: dict[tuple[int, ...], int], kernel: int) -> onnx.ModelProto:
    pad = kernel // 2
    center = (kernel * kernel) // 2

    overrides = [(w, lut[w]) for w in lut if w[center] != lut[w]]
    n_ovr = len(overrides)

    # --- Identity path: 1×1 conv that copies each channel ---
    id_w = np.zeros((CHANNELS, CHANNELS, 1, 1), dtype=np.float32)
    for c in range(CHANNELS):
        id_w[c, c, 0, 0] = 1.0
    id_bias = np.full((CHANNELS,), -0.5, dtype=np.float32)

    id_W = make_const("id_W", id_w)
    id_B = make_const("id_B", id_bias)
    id_conv = helper.make_node(
        "Conv", [INPUT_NAME, "id_W", "id_B"], ["id_raw"],
        kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], name="id_conv",
    )
    id_relu = helper.make_node("Relu", ["id_raw"], ["id_out"], name="id_relu")

    if n_ovr == 0:
        return make_model(
            [id_conv, id_relu,
             helper.make_node("Identity", ["id_out"], [OUTPUT_NAME], name="copy")],
            [id_W, id_B],
            doc=f"pure identity (kernel={kernel})",
        )

    # --- Override path ---
    ovr_w = np.zeros((n_ovr, CHANNELS, kernel, kernel), dtype=np.float32)
    ovr_bias = np.zeros((n_ovr,), dtype=np.float32)
    router = np.zeros((CHANNELS, n_ovr, 1, 1), dtype=np.float32)

    for idx, (win, new_color) in enumerate(overrides):
        ovr_w[idx] = _one_hot_window(win, kernel)
        n_valid = sum(1 for v in win if 0 <= v < CHANNELS)
        ovr_bias[idx] = -(n_valid - 0.5) if n_valid > 0 else -0.5

        old_color = win[center]
        if 0 <= new_color < CHANNELS:
            router[new_color, idx, 0, 0] = 2.0
        if 0 <= old_color < CHANNELS:
            router[old_color, idx, 0, 0] = -2.0

    ovr_W = make_const("ovr_W", ovr_w)
    ovr_B = make_const("ovr_B", ovr_bias)
    rtr_W = make_const("rtr_W", router)

    ovr_conv = helper.make_node(
        "Conv", [INPUT_NAME, "ovr_W", "ovr_B"], ["ovr_raw"],
        kernel_shape=[kernel, kernel], strides=[1, 1],
        pads=[pad, pad, pad, pad], name="ovr_conv",
    )
    ovr_relu = helper.make_node("Relu", ["ovr_raw"], ["ovr_det"], name="ovr_relu")
    ovr_route = helper.make_node(
        "Conv", ["ovr_det", "rtr_W"], ["ovr_out"],
        kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], name="ovr_route",
    )

    add_node = helper.make_node("Add", ["id_out", "ovr_out"], [OUTPUT_NAME], name="combine")

    return make_model(
        [id_conv, id_relu, ovr_conv, ovr_relu, ovr_route, add_node],
        [id_W, id_B, ovr_W, ovr_B, rtr_W],
        doc=f"identity+override: kernel={kernel}, overrides={n_ovr}",
    )


def can_fit(task: dict, kernel: int, limit_bytes: int = 1_509_949) -> tuple[bool, int]:
    lut = extract_lut(task, kernel)
    if lut is None:
        return False, -1
    center = (kernel * kernel) // 2
    n_ovr = sum(1 for w in lut if w[center] != lut[w])
    est = 5000 + n_ovr * (CHANNELS * kernel * kernel + 1 + CHANNELS) * 4
    return est < limit_bytes, n_ovr


__all__ = ["can_fit", "compile_identity_override"]
