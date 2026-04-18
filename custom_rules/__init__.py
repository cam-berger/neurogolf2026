"""Registry of per-task hand-coded local-rule generators.

Each file `task{NNN}.py` in this directory should expose:

    def generate(task: dict, features: dict) -> onnx.ModelProto | None:
        ...

Files are auto-discovered on import. `LocalRuleGenerator` checks this
registry before falling back to the gradient-based training loop.
"""

from __future__ import annotations

import re
from importlib import import_module
from pathlib import Path
from typing import Callable, Protocol

import onnx


class _GenerateFn(Protocol):
    def __call__(self, task: dict, features: dict) -> onnx.ModelProto | None: ...


_registry: dict[int, _GenerateFn] = {}


def register(task_id: int, fn: _GenerateFn) -> None:
    _registry[task_id] = fn


def get(task_id: int) -> _GenerateFn | None:
    return _registry.get(task_id)


def _autodiscover() -> None:
    pat = re.compile(r"task(\d{3})\.py$")
    for p in sorted(Path(__file__).parent.glob("task*.py")):
        m = pat.match(p.name)
        if not m:
            continue
        module = import_module(f"custom_rules.{p.stem}")
        if hasattr(module, "generate"):
            register(int(m.group(1)), module.generate)


_autodiscover()


__all__ = ["get", "register"]
