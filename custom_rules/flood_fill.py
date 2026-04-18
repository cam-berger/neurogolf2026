"""Flood-fill primitive: detect cells enclosed by walls, fill with a color.

Uses iterative 3x3 max-propagation conv. ~30 iterations guarantees convergence
for 30x30 grids. Outputs cells that are 0 AND not reachable from the grid edges.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper

from generators.base import INPUT_NAME, OUTPUT_NAME, make_const, make_int_const, make_model
from pipeline.loader import HEIGHT, WIDTH, CHANNELS


def _build_reachable(input_name: str, wall_color: int | None, n_iter: int = 30) -> tuple[list, list, str]:
    """Build nodes that compute 'reachable from border' mask [1, 1, 30, 30].

    Returns (nodes, inits, reachable_tensor_name).
    """
    nodes = []
    inits = []

    # Step 1: compute wall_mask = sum of "wall" channels (or any nonzero channel)
    if wall_color is None:
        # Any non-zero cell is a wall. Use ReduceSum over channels 1..9 (skip channel 0 which is "color 0")
        # Actually: color 0 means the cell has channel 0 active. In one-hot, any channel active = cell has a color.
        # A cell is "empty" (color 0) if ONLY channel 0 is active.
        # So wall_mask = 1 if any channel != 0 is active.
        # We want: sum over channels 1..9 of input > 0
        # Use ReduceSum with axes=[1] on channels 1..9. Need to Slice channels first.
        starts = make_int_const("ch_starts", [1])
        ends = make_int_const("ch_ends", [CHANNELS])
        axes = make_int_const("ch_axes", [1])
        steps = make_int_const("ch_steps", [1])
        inits.extend([starts, ends, axes, steps])
        nodes.append(helper.make_node("Slice", [input_name, "ch_starts", "ch_ends", "ch_axes", "ch_steps"],
                                       ["nonzero_channels"], name="slice_channels"))
        nodes.append(helper.make_node("ReduceSum", ["nonzero_channels"], ["wall_mask_pre"],
                                       axes=[1], keepdims=1, name="reduce_channels"))
    else:
        # Only a specific color is a wall
        # Slice channel wall_color
        starts = make_int_const("wc_starts", [wall_color])
        ends = make_int_const("wc_ends", [wall_color + 1])
        axes = make_int_const("wc_axes", [1])
        steps = make_int_const("wc_steps", [1])
        inits.extend([starts, ends, axes, steps])
        nodes.append(helper.make_node("Slice", [input_name, "wc_starts", "wc_ends", "wc_axes", "wc_steps"],
                                       ["wall_mask_pre"], name="slice_wall"))

    # Threshold wall_mask to binary {0, 1}. Opset 10 Clip uses attributes.
    nodes.append(helper.make_node("Clip", ["wall_mask_pre"], ["wall_mask"],
                                   min=0.0, max=1.0, name="clip_wall"))

    # not_wall = 1 - wall_mask
    one_c = make_const("one_tensor", np.ones((1, 1, HEIGHT, WIDTH), dtype=np.float32))
    inits.append(one_c)
    nodes.append(helper.make_node("Sub", ["one_tensor", "wall_mask"], ["not_wall"], name="sub_not_wall"))

    # Initial reachable: border of the 30x30 tensor
    border = np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32)
    border[0, 0, 0, :] = 1.0
    border[0, 0, -1, :] = 1.0
    border[0, 0, :, 0] = 1.0
    border[0, 0, :, -1] = 1.0
    border_c = make_const("border_init", border)
    inits.append(border_c)

    # reachable = border * not_wall  (can't propagate through walls, so zero out border walls)
    nodes.append(helper.make_node("Mul", ["border_init", "not_wall"], ["reach_0"], name="init_reach"))

    # Convolution kernel for 3x3 "any neighbor active"
    conv_w = np.ones((1, 1, 3, 3), dtype=np.float32)
    conv_c = make_const("conv3x3", conv_w)
    inits.append(conv_c)

    # Iterate: new_reach = clip((old_reach * conv3x3) * not_wall, 0, 1)
    # Actually: new_reach = max(old_reach, (conv(old_reach) * not_wall) clipped to [0, 1])
    # Simpler: new_reach = clip(old_reach + conv(old_reach) * not_wall, 0, 1)
    cur = "reach_0"
    for step in range(n_iter):
        next_name = f"reach_{step+1}"
        # conv: compute sum of 3x3 neighborhood
        conv_out = f"conv_out_{step}"
        nodes.append(helper.make_node("Conv", [cur, "conv3x3"], [conv_out],
                                       kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1],
                                       name=f"conv_{step}"))
        # Gate by not_wall
        gated = f"gated_{step}"
        nodes.append(helper.make_node("Mul", [conv_out, "not_wall"], [gated], name=f"gate_{step}"))
        # Add to current reachable
        summed = f"summed_{step}"
        nodes.append(helper.make_node("Add", [cur, gated], [summed], name=f"add_{step}"))
        # Clip to [0, 1]
        nodes.append(helper.make_node("Clip", [summed], [next_name], min=0.0, max=1.0, name=f"clip_{step}"))
        cur = next_name

    return nodes, inits, cur, "not_wall"


def build_fill_enclosed(fill_color: int, wall_color: int | None = None,
                         n_iter: int = 30) -> onnx.ModelProto:
    """Build ONNX: identity + fill enclosed 0-cells with `fill_color`.

    If wall_color is given, only that color counts as wall. Otherwise any non-zero.
    """
    nodes, inits, reach_name, not_wall_name = _build_reachable(INPUT_NAME, wall_color, n_iter)

    # enclosed = not_wall * (1 - reachable)
    one_c2 = make_const("one_scalar", np.array([1.0], dtype=np.float32))
    inits.append(one_c2)
    nodes.append(helper.make_node("Sub", ["one_scalar", reach_name], ["not_reach"], name="sub_not_reach"))
    nodes.append(helper.make_node("Mul", [not_wall_name, "not_reach"], ["enclosed"], name="mul_enclosed"))

    # Identity path: copy input via 1x1 conv + bias to stay above 0.5 threshold
    id_w = np.zeros((CHANNELS, CHANNELS, 1, 1), dtype=np.float32)
    for c in range(CHANNELS):
        id_w[c, c, 0, 0] = 1.0
    id_bias = np.full((CHANNELS,), -0.5, dtype=np.float32)
    inits.append(make_const("id_W", id_w))
    inits.append(make_const("id_B", id_bias))
    nodes.append(helper.make_node("Conv", [INPUT_NAME, "id_W", "id_B"], ["id_raw"],
                                   kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], name="id_conv"))
    nodes.append(helper.make_node("Relu", ["id_raw"], ["id_out"], name="id_relu"))

    # Fill path: add fill_color channel at enclosed cells, and suppress channel 0 at enclosed cells
    # enclosed is [1, 1, 30, 30]. Need to create overlay [1, 10, 30, 30]:
    #   channel fill_color: +2.0 at enclosed
    #   channel 0: -2.0 at enclosed (if 0 was active, override)
    #
    # We use a 1x1 conv: take enclosed [1, 1, 30, 30] → [1, 10, 30, 30]
    overlay_w = np.zeros((CHANNELS, 1, 1, 1), dtype=np.float32)
    overlay_w[fill_color, 0, 0, 0] = 2.0
    overlay_w[0, 0, 0, 0] = -2.0  # suppress channel 0 at enclosed
    inits.append(make_const("overlay_W", overlay_w))
    nodes.append(helper.make_node("Conv", ["enclosed", "overlay_W"], ["fill_overlay"],
                                   kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], name="overlay_conv"))

    # Combine: output = identity + fill_overlay
    nodes.append(helper.make_node("Add", ["id_out", "fill_overlay"], [OUTPUT_NAME], name="combine"))

    return make_model(nodes, inits, doc=f"flood fill enclosed with color {fill_color}, wall_color={wall_color}")


__all__ = ["build_fill_enclosed"]
