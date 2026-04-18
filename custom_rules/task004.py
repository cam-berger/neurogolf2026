"""Task 004 — rubber-stamp LUT baseline while the real rule is being authored.

Remove this file (or replace its body) once a compressed hand-rule is ready.
"""

from __future__ import annotations

import onnx

from custom_rules.lut import compile_lut_to_onnx, extract_lut


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    lut = extract_lut(task, kernel=3)
    if lut is None:
        return None
    return compile_lut_to_onnx(lut, kernel=3)
