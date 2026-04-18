"""LUT helpers for hand-written local-rule generators.

`extract_lut` walks every (input, output) pair in the full pool and builds
a `{window_tuple -> output_color}` dict. The window tuple is an int tuple
of length k*k; cells outside the grid are encoded as the sentinel -1.

`compile_lut_to_onnx` turns a LUT into an analytical two-layer conv network:

    Layer 1: Conv(k, k, in=10, out=H, bias) — one filter per unique window.
             Weights are the one-hot window tensors; bias is -(k*k - 0.5)
             so that only an exact match survives ReLU.
    Layer 2: Conv(1, 1, in=H, out=10, no bias) — routes detector `w` to
             its output color with weight 2.0 (0.5 * 2.0 = 1.0 > 0).

Outside-grid cells have all-zero channels, so layer-1 convolution yields 0;
the negative bias drives the logit < 0; ReLU clips it to 0; layer-2 output
stays 0. That matches the official scoring convention.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import helper

from generators.base import (
    INPUT_NAME,
    OUTPUT_NAME,
    make_const,
    make_model,
)
from pipeline.loader import CHANNELS, get_all_pairs


def extract_lut(task: dict, kernel: int) -> dict[tuple[int, ...], int] | None:
    """Build the observed (k*k window) -> output color table from the full pool.

    Returns None if the mapping is inconsistent (two inputs sharing the
    same window map to different outputs) — in that case the task is not a
    pure local rule of this kernel size.
    """
    if kernel % 2 != 1 or kernel < 1:
        raise ValueError("kernel must be a positive odd integer")
    r = kernel // 2
    lut: dict[tuple[int, ...], int] = {}
    for inp, out in get_all_pairs(task):
        i = np.asarray(inp, dtype=np.int64)
        o = np.asarray(out, dtype=np.int64)
        if i.shape != o.shape:
            return None
        h, w = i.shape
        padded = np.pad(i, r, mode="constant", constant_values=-1)
        for y in range(h):
            for x in range(w):
                window = tuple(int(v) for v in padded[y:y + kernel, x:x + kernel].flatten())
                color = int(o[y, x])
                if window in lut:
                    if lut[window] != color:
                        return None
                else:
                    lut[window] = color
    return lut


def _one_hot_window(window: tuple[int, ...], kernel: int) -> np.ndarray:
    """Materialize a k*k window tuple as a [10, k, k] float32 filter.

    - In-grid cell of color c: channel c gets +1.0, others 0.
    - Out-of-grid cell (value -1): all channels get -1.0, so any input color
      at that position drags the conv sum below the match threshold and
      prevents the detector from firing spuriously when -1 slots happen to
      sit over interior grid cells.
    """
    w = np.zeros((CHANNELS, kernel, kernel), dtype=np.float32)
    for idx, val in enumerate(window):
        r, c = divmod(idx, kernel)
        if 0 <= val < CHANNELS:
            w[val, r, c] = 1.0
        else:  # sentinel: outside grid
            w[:, r, c] = -1.0
    return w


def compile_lut_to_onnx(lut: dict[tuple[int, ...], int], kernel: int) -> onnx.ModelProto:
    """Compile a LUT into a two-conv analytical network.

    Score note: param count is ~H * (10*k*k + 10) with H = len(lut). For
    tasks with a few hundred unique windows, cost is dominated by MACs on
    the 30x30 spatial grid. See the module docstring for boundary handling.
    """
    if kernel % 2 != 1 or kernel < 1:
        raise ValueError("kernel must be a positive odd integer")
    if not lut:
        raise ValueError("LUT is empty")

    windows = list(lut.keys())
    h = len(windows)
    pad = kernel // 2

    # Layer 1: detector filters. Bias is calibrated per detector: an exact
    # match yields a conv sum equal to the number of non-(-1) cells in the
    # window (because -1 cells translate to all-zero filter positions and
    # therefore contribute zero to the dot product). Threshold just below
    # that count so only exact matches survive ReLU.
    detector_w = np.zeros((h, CHANNELS, kernel, kernel), dtype=np.float32)
    detector_bias = np.zeros((h,), dtype=np.float32)
    for idx, win in enumerate(windows):
        detector_w[idx] = _one_hot_window(win, kernel)
        n_valid = sum(1 for v in win if 0 <= v < CHANNELS)
        detector_bias[idx] = -(n_valid - 0.5) if n_valid > 0 else -0.5

    # Layer 2: router from detectors back to 10 color channels.
    router = np.zeros((CHANNELS, h, 1, 1), dtype=np.float32)
    for idx, win in enumerate(windows):
        color = lut[win]
        if 0 <= color < CHANNELS:
            router[color, idx, 0, 0] = 2.0

    w1 = make_const("W1", detector_w)
    b1 = make_const("B1", detector_bias)
    w2 = make_const("W2", router)

    conv1 = helper.make_node(
        "Conv",
        [INPUT_NAME, "W1", "B1"],
        ["detect_raw"],
        name="lut_detect",
        kernel_shape=[kernel, kernel],
        strides=[1, 1],
        pads=[pad, pad, pad, pad],
    )
    relu = helper.make_node("Relu", ["detect_raw"], ["detect"], name="lut_relu")
    conv2 = helper.make_node(
        "Conv",
        ["detect", "W2"],
        [OUTPUT_NAME],
        name="lut_route",
        kernel_shape=[1, 1],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )

    return make_model(
        [conv1, relu, conv2],
        [w1, b1, w2],
        doc=f"LUT-compiled local rule: kernel={kernel}, unique windows={h}",
    )


__all__ = ["compile_lut_to_onnx", "extract_lut"]
