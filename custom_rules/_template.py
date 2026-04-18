"""Template for hand-coded local-rule generators.

Copy this to `task{NNN}.py` and fill in `generate(...)`. The file name is
how the registry discovers your rule — the digits after `task` become the
task id that `LocalRuleGenerator` will route to you.

Three common patterns:

1. Rubber-stamp the observed LUT:

       from custom_rules.lut import compile_lut_to_onnx, extract_lut

       def generate(task, features):
           lut = extract_lut(task, kernel=3)
           if lut is None:
               return None
           return compile_lut_to_onnx(lut, kernel=3)

2. Compress the LUT to a smaller table before compiling (drop redundant
   windows, merge symmetries, etc.) — same helpers, just prune `lut` first.

3. Hand-author conv weights for a specific rule (Game-of-Life-style
   thresholds, parity checks, etc.) using `generators.base.make_model`.
"""

from __future__ import annotations

import onnx


def generate(task: dict, features: dict) -> onnx.ModelProto | None:
    raise NotImplementedError("fill me in")
