"""Task 052 — deep trainer 5x5."""

from __future__ import annotations

import onnx

from custom_rules.deep_trainer import train_deep


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return train_deep(task, kernel=5, hidden=32, n_steps=3000, n_restarts=2, verbose=False)
