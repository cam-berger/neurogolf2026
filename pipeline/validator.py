"""ONNX correctness checking, cost scoring, and constraint validation.

Cost calculation delegates to `onnx_tool` because that is what the official
Kaggle scorer (see `neurogolf_utils.score_network`) uses — matching its
output matters more than re-deriving it manually.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime

from pipeline.loader import GRID_SHAPE, encode_grid, get_all_pairs

_FORBIDDEN_OPS = {"LOOP", "SCAN", "NONZERO", "UNIQUE", "SCRIPT", "FUNCTION"}
_FILESIZE_LIMIT_BYTES = int(1.44 * 1024 * 1024)


def _session(onnx_path: str) -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def _input_name(session: onnxruntime.InferenceSession) -> str:
    return session.get_inputs()[0].name


def _output_name(session: onnxruntime.InferenceSession) -> str:
    return session.get_outputs()[0].name


def run_network(onnx_path: str, input_tensor: np.ndarray) -> np.ndarray:
    """Run an ONNX model on a pre-encoded [1, 10, 30, 30] tensor."""
    sess = _session(onnx_path)
    result = sess.run([_output_name(sess)], {_input_name(sess): input_tensor.astype(np.float32)})
    return result[0]


def _encode_output_expected(grid: list[list[int]]) -> np.ndarray:
    """Encode an expected output grid into the binary one-hot tensor the scorer compares against."""
    return encode_grid(grid)


def check_correctness(onnx_path: str, task: dict) -> dict:
    """Run the model on every pair and compare using official semantics.

    Correctness criterion (matches `verify_subset` in neurogolf_utils):
    thresholded output `(out > 0.0).astype(float)` must equal the expected
    one-hot tensor via `np.array_equal`.
    """
    sess = _session(onnx_path)
    in_name, out_name = _input_name(sess), _output_name(sess)

    pairs = get_all_pairs(task)
    failures: list[int] = []
    n_correct = 0

    for idx, (input_grid, output_grid) in enumerate(pairs):
        x = encode_grid(input_grid)
        try:
            raw = sess.run([out_name], {in_name: x})[0]
        except Exception:
            failures.append(idx)
            continue
        predicted = (raw > 0.0).astype(np.float32)
        expected = _encode_output_expected(output_grid)
        if predicted.shape == expected.shape and np.array_equal(predicted, expected):
            n_correct += 1
        else:
            failures.append(idx)

    return {
        "correct": len(failures) == 0,
        "n_pairs": len(pairs),
        "n_correct": n_correct,
        "failures": failures,
    }


def compute_cost(onnx_path: str, input_shape: tuple = GRID_SHAPE) -> dict:
    """Compute MACs + memory + params using onnx_tool (official scorer path)."""
    import onnx_tool  # lazy import; avoids hard dep at module load

    model = onnx_tool.loadmodel(onnx_path, {"verbose": False})
    g = model.graph
    g.graph_reorder_nodes()
    g.shape_infer(None)
    g.profile()
    if not g.valid_profile:
        return {"valid": False, "reason": "onnx_tool could not profile the graph"}

    macs = int(sum(g.macs))
    memory_bytes = int(g.memory)
    n_params = int(g.params)
    cost = macs + memory_bytes + n_params
    score = max(1.0, 25.0 - math.log(cost)) if cost > 0 else 25.0

    return {
        "valid": True,
        "mac_ops": macs,
        "memory_bytes": memory_bytes,
        "n_params": n_params,
        "cost": cost,
        "score": score,
    }


def _iter_shapes(model: onnx.ModelProto):
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        yield vi.name, vi.type.tensor_type.shape


def validate_constraints(onnx_path: str) -> dict:
    """Check file-size / forbidden-op / static-shape constraints."""
    violations: list[str] = []
    path = Path(onnx_path)

    if not path.is_file():
        return {"valid": False, "violations": [f"file not found: {onnx_path}"]}

    size = path.stat().st_size
    if size > _FILESIZE_LIMIT_BYTES:
        violations.append(f"file size {size}B exceeds limit {_FILESIZE_LIMIT_BYTES}B")

    model = onnx.load(onnx_path)
    for node in model.graph.node:
        if node.op_type.upper() in _FORBIDDEN_OPS:
            violations.append(f"forbidden op: {node.op_type} (node {node.name or '<unnamed>'})")

    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception as e:  # noqa: BLE001
        violations.append(f"shape inference failed: {e}")
        inferred = model

    for name, shape in _iter_shapes(inferred):
        for dim in shape.dim:
            if dim.HasField("dim_param") or (dim.HasField("dim_value") and dim.dim_value <= 0):
                violations.append(f"dynamic/invalid shape on tensor '{name}'")
                break

    return {"valid": len(violations) == 0, "violations": violations, "file_size": size}


def format_cost(cost: dict) -> str:
    if not cost.get("valid"):
        return f"invalid ({cost.get('reason', 'unknown')})"
    return (
        f"macs={cost['mac_ops']} mem={cost['memory_bytes']}B "
        f"params={cost['n_params']} score={cost['score']:.3f}"
    )


__all__ = [
    "check_correctness",
    "compute_cost",
    "format_cost",
    "run_network",
    "validate_constraints",
]
