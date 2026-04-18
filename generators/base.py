"""Abstract base class + shared helpers for ONNX network generators.

All generators emit models with:
- Opset 10 / IR version 10 (matches `neurogolf_utils`)
- Static input/output shape [1, 10, 30, 30] named "input" / "output"
- No forbidden ops
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import onnx
from onnx import TensorProto, helper

from pipeline.loader import GRID_SHAPE

IR_VERSION = 10
OPSET = [helper.make_opsetid("", 10)]
INPUT_NAME = "input"
OUTPUT_NAME = "output"
DTYPE = TensorProto.FLOAT


def make_input() -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(INPUT_NAME, DTYPE, list(GRID_SHAPE))


def make_output() -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(OUTPUT_NAME, DTYPE, list(GRID_SHAPE))


def make_model(nodes, initializers, doc: str = "") -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes,
        name="graph",
        inputs=[make_input()],
        outputs=[make_output()],
        initializer=initializers,
    )
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET, producer_name="arc_neurogolf")
    if doc:
        model.doc_string = doc
    onnx.checker.check_model(model)
    return model


def make_const(name: str, array: np.ndarray, dtype=DTYPE) -> onnx.TensorProto:
    return helper.make_tensor(
        name=name,
        data_type=dtype,
        dims=list(array.shape),
        vals=array.flatten().tolist(),
    )


def make_int_const(name: str, values: list[int]) -> onnx.TensorProto:
    arr = np.asarray(values, dtype=np.int64)
    return helper.make_tensor(
        name=name,
        data_type=TensorProto.INT64,
        dims=list(arr.shape),
        vals=arr.flatten().tolist(),
    )


def task_shape_signature(task: dict) -> tuple | None:
    """If every (input, output) pair shares the same shapes, return the tuple
    ((H_in, W_in), (H_out, W_out)); otherwise None.
    """
    ish = osh = None
    for split in ("train", "test", "arc_gen"):
        for ex in task.get(split, []):
            i = (len(ex["input"]), len(ex["input"][0]) if ex["input"] else 0)
            o = (len(ex["output"]), len(ex["output"][0]) if ex["output"] else 0)
            if ish is None:
                ish, osh = i, o
            elif (i, o) != (ish, osh):
                return None
    if ish is None:
        return None
    return ish, osh


class NetworkGenerator(ABC):
    family: str = "abstract"

    @abstractmethod
    def can_generate(self, task: dict, features: dict) -> bool:
        """Return True if this generator can emit a correct network for this task."""

    @abstractmethod
    def generate(self, task: dict, features: dict) -> onnx.ModelProto | None:
        """Emit a minimal ONNX model; return None if generation fails."""

    def save(self, model: onnx.ModelProto, path: str) -> None:
        onnx.save(model, path)


__all__ = [
    "DTYPE",
    "INPUT_NAME",
    "IR_VERSION",
    "NetworkGenerator",
    "OPSET",
    "OUTPUT_NAME",
    "make_const",
    "make_input",
    "make_int_const",
    "make_model",
    "make_output",
    "task_shape_signature",
]
