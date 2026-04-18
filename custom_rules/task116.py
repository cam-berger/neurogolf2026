"""Task 116 — reflect_concat."""

from __future__ import annotations
import onnx
from custom_rules.reflect_concat import build_vflip_up
from pipeline.loader import get_all_pairs


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    sizes = sorted(set([len(i) for i, _ in get_all_pairs(task)]))
    return build_vflip_up(sizes)
