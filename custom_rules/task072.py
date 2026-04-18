"""Task 72 — split+xor with color 3."""

from __future__ import annotations
import onnx
from custom_rules.split_combine import build_split_combine


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return build_split_combine("horizontal", 6, 1, "xor", 3, 6, 5)
