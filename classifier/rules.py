"""Rule-based family classifier built on top of extracted features."""

from __future__ import annotations

from classifier.families import TransformFamily
from classifier.features import extract_features
from pipeline.loader import load_task


def classify_task(features: dict) -> TransformFamily:
    """Resolve the transformation family from feature flags.

    Checks run in priority order — the first match wins.
    """
    if features.get("_empty"):
        return TransformFamily.UNKNOWN

    if features["is_identity"]:
        return TransformFamily.IDENTITY
    if features["is_rot90"]:
        return TransformFamily.ROT90
    if features["is_rot180"]:
        return TransformFamily.ROT180
    if features["is_rot270"]:
        return TransformFamily.ROT270
    if features["is_flip_h"]:
        return TransformFamily.FLIP_H
    if features["is_flip_v"]:
        return TransformFamily.FLIP_V
    if features["is_transpose"]:
        return TransformFamily.TRANSPOSE
    if features["is_color_permutation"] and features["output_shape_eq_input"]:
        return TransformFamily.COLOR_REMAP
    if features["output_is_input_scaled"] and features["scale_factor_h"] >= 2 and features["scale_factor_w"] >= 2:
        return TransformFamily.SCALE_UP

    if features["output_shape_eq_input"]:
        k = features["max_local_context_needed"]
        if k <= 3:
            return TransformFamily.LOCAL_RULE_3x3
        if k <= 5:
            return TransformFamily.LOCAL_RULE_5x5

    return TransformFamily.UNKNOWN


def classify_all_tasks(task_ids) -> dict[int, TransformFamily]:
    """Classify a batch of tasks. Missing task files are skipped silently."""
    results: dict[int, TransformFamily] = {}
    for tid in task_ids:
        try:
            task = load_task(tid)
        except FileNotFoundError:
            continue
        results[tid] = classify_task(extract_features(task))
    return results


__all__ = ["classify_task", "classify_all_tasks"]
