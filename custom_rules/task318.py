"""Task 318 — split+or with color 3."""

from __future__ import annotations
import onnx
from custom_rules.split_combine import build_split_combine


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return build_split_combine("horizontal", 4, 1, "or", 3, 4, 4)
