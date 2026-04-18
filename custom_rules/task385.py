"""Task 385 — vertical mirror (bottom)."""

from __future__ import annotations
import onnx
from custom_rules.vertical_mirror import build_vertical_mirror


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return build_vertical_mirror("bottom")
