"""Task 249 — tile x2 horizontal."""
from __future__ import annotations
import onnx
from custom_rules.tile_generators import build_task_249


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return build_task_249(task)
