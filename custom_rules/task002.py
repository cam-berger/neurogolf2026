"""Task 002: fill 0-cells enclosed by non-zero walls with color 4."""
from __future__ import annotations
import onnx
from custom_rules.flood_fill import build_fill_enclosed


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return build_fill_enclosed(fill_color=4, wall_color=None, n_iter=30)
