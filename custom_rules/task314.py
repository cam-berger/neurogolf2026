"""Task 314 — identity+override 7x7."""

from __future__ import annotations
import onnx
from custom_rules.identity_override import compile_identity_override
from custom_rules.lut import extract_lut


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    lut = extract_lut(task, kernel=7)
    if lut is None: return None
    return compile_identity_override(lut, kernel=7)
