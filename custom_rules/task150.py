"""Task 150 — dynamic-shape geometric generator."""

from __future__ import annotations

import onnx

from custom_rules.dynamic_geometric import build_for_task


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return build_for_task(150, task)
