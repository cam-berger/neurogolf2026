"""Task 389 — deep trainer."""

from __future__ import annotations
import onnx
from custom_rules.deep_trainer import train_deep


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    return train_deep(task, kernel=5, hidden=24, n_steps=1500, n_restarts=1, verbose=False)
