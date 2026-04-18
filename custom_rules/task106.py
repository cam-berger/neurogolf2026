"""Task 106 — rotation quad."""
from __future__ import annotations
import onnx
from custom_rules.rotation_quad import build_rotation_quad
from pipeline.loader import get_all_pairs


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    sizes = sorted(set(len(i) for i, _ in get_all_pairs(task)))
    return build_rotation_quad(sizes)
