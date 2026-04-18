"""Task 6 — split+and with color 2."""

from __future__ import annotations
import onnx
from custom_rules.split_combine import build_split_combine


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return build_split_combine("vertical", 3, 1, "and", 2, 3, 3)
