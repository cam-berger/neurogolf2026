"""Task 142 — quad mirror (h=3, w=3)."""
from __future__ import annotations
import onnx
from custom_rules.quad_mirror import build_quad_mirror


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return build_quad_mirror(3, 3)
