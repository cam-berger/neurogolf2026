"""Task loading and grid <-> tensor conversion.

Encoding/decoding match the official `neurogolf_utils` semantics:
- Encoded tensor is float32 [1, 10, 30, 30]; cell (r,c) with color k sets
  channel k to 1.0, all others to 0.0. Cells outside the grid are zero
  across every channel.
- Decoding thresholds the output with (> 0.0) so that each cell is tagged
  by the set of channels that fired. 0 channels -> 10 ("no color"),
  1 channel -> that color, 2+ channels -> 11 ("too many colors").
- Trailing "no color" cells are stripped per row, then trailing empty rows
  are stripped — so the network output implicitly encodes grid dimensions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

BATCH_SIZE, CHANNELS, HEIGHT, WIDTH = 1, 10, 30, 30
GRID_SHAPE = (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _task_path(task_id: int) -> Path:
    return PROJECT_ROOT / f"task{task_id:03d}.json"


def load_task(task_id: int) -> dict:
    """Load task JSON and normalize the arc-gen key to arc_gen."""
    with open(_task_path(task_id)) as f:
        raw = json.load(f)
    return {
        "train": raw.get("train", []),
        "test": raw.get("test", []),
        "arc_gen": raw.get("arc-gen", []),
    }


def encode_grid(grid: list[list[int]]) -> np.ndarray:
    """One-hot encode a variable-size grid into a [1, 10, 30, 30] float32 tensor."""
    tensor = np.zeros(GRID_SHAPE, dtype=np.float32)
    for r, row in enumerate(grid):
        if r >= HEIGHT:
            break
        for c, color in enumerate(row):
            if c >= WIDTH:
                break
            if 0 <= color < CHANNELS:
                tensor[0, color, r, c] = 1.0
    return tensor


def decode_grid(tensor: np.ndarray, height: int | None = None, width: int | None = None) -> list[list[int]]:
    """Decode a [1, 10, 30, 30] tensor back to a grid, matching official semantics.

    Args:
        tensor: float32 tensor. Will be thresholded with (> 0.0).
        height / width: optional crop. If None, returns the full decoded grid
            (which self-trims trailing "no color" cells/rows, as the official
            scorer does).
    """
    mask = (tensor > 0.0).astype(np.int32)
    _, channels, full_h, full_w = mask.shape
    h = full_h if height is None else min(height, full_h)
    w = full_w if width is None else min(width, full_w)

    grid: list[list[int]] = []
    for row in range(h):
        cells: list[int] = []
        for col in range(w):
            colors = [c for c in range(channels) if mask[0, c, row, col] == 1]
            if len(colors) == 1:
                cells.append(colors[0])
            elif len(colors) == 0:
                cells.append(10)  # sentinel: "no color"
            else:
                cells.append(11)  # sentinel: "too many colors"
        if height is None:
            while cells and cells[-1] == 10:
                cells.pop()
        grid.append(cells)

    if height is None:
        while grid and not grid[-1]:
            grid.pop()
    return grid


def get_all_pairs(task: dict) -> list[tuple[list[list[int]], list[list[int]]]]:
    """Flatten train + test + arc_gen into a list of (input, output) pairs."""
    pairs: list[tuple[Any, Any]] = []
    for split in ("train", "test", "arc_gen"):
        for ex in task.get(split, []):
            pairs.append((ex["input"], ex["output"]))
    return pairs
