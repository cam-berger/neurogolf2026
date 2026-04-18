"""Task 236 — split+xor with color 3."""

from __future__ import annotations
import onnx
from custom_rules.split_combine import build_split_combine


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return build_split_combine("horizontal", 4, 1, "xor", 3, 4, 4)
