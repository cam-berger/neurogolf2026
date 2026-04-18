"""Task 326 — crop generator."""

from __future__ import annotations

import onnx

from custom_rules.crop_generators import build_task_326


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return build_task_326(task)
