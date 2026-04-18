"""Data pipeline: task loading, grid encoding, ONNX validation.

Validator helpers (check_correctness / compute_cost / validate_constraints /
run_network) are importable from `pipeline.validator` directly but are not
re-exported here, so the package stays usable when `onnx` / `onnxruntime`
are not installed (Phase 1 only needs numpy).
"""

from pipeline.loader import (
    BATCH_SIZE,
    CHANNELS,
    GRID_SHAPE,
    HEIGHT,
    PROJECT_ROOT,
    WIDTH,
    decode_grid,
    encode_grid,
    get_all_pairs,
    load_task,
)

__all__ = [
    "BATCH_SIZE",
    "CHANNELS",
    "GRID_SHAPE",
    "HEIGHT",
    "PROJECT_ROOT",
    "WIDTH",
    "decode_grid",
    "encode_grid",
    "get_all_pairs",
    "load_task",
]
