"""Feature extraction for ARC tasks.

All features are derived from the `train` pairs. A boolean feature is True
only if it holds for every train pair — consistency across pairs is the
whole point of the classifier.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def _as_array(grid: list[list[int]]) -> np.ndarray:
    return np.array(grid, dtype=np.int64)


def _train_arrays(task: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(_as_array(p["input"]), _as_array(p["output"])) for p in task.get("train", [])]


def _all_arrays(task: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every visible pair (train + test + arc_gen), for consistency checks
    that need to generalize beyond the 3-5 train examples."""
    out = []
    for split in ("train", "test", "arc_gen"):
        for p in task.get(split, []):
            out.append((_as_array(p["input"]), _as_array(p["output"])))
    return out


# ---- Geometric checks --------------------------------------------------------

def _all(pairs, predicate) -> bool:
    return bool(pairs) and all(predicate(i, o) for i, o in pairs)


def _is_identity(pairs) -> bool:
    return _all(pairs, lambda i, o: i.shape == o.shape and np.array_equal(i, o))


def _is_rot90(pairs) -> bool:  # clockwise
    return _all(pairs, lambda i, o: o.shape == (i.shape[1], i.shape[0]) and np.array_equal(np.rot90(i, k=-1), o))


def _is_rot180(pairs) -> bool:
    return _all(pairs, lambda i, o: o.shape == i.shape and np.array_equal(np.rot90(i, k=2), o))


def _is_rot270(pairs) -> bool:  # 90 CCW
    return _all(pairs, lambda i, o: o.shape == (i.shape[1], i.shape[0]) and np.array_equal(np.rot90(i, k=1), o))


def _is_flip_h(pairs) -> bool:
    return _all(pairs, lambda i, o: o.shape == i.shape and np.array_equal(np.fliplr(i), o))


def _is_flip_v(pairs) -> bool:
    return _all(pairs, lambda i, o: o.shape == i.shape and np.array_equal(np.flipud(i), o))


def _is_transpose(pairs) -> bool:
    return _all(pairs, lambda i, o: o.shape == (i.shape[1], i.shape[0]) and np.array_equal(i.T, o))


# ---- Color-permutation check -------------------------------------------------

def _color_permutation(pairs) -> dict | None:
    """Return a consistent bijective color map across all pairs, or None."""
    if not pairs:
        return None
    mapping: dict[int, int] = {}
    for i, o in pairs:
        if i.shape != o.shape:
            return None
        for a, b in zip(i.flatten(), o.flatten()):
            a, b = int(a), int(b)
            if a in mapping:
                if mapping[a] != b:
                    return None
            else:
                mapping[a] = b
    # injectivity on the observed range
    if len(set(mapping.values())) != len(mapping):
        return None
    return mapping


# ---- Scale detection ---------------------------------------------------------

def _detect_scale(pairs) -> tuple[bool, float, float]:
    """Detect an integer nearest-neighbor upscale that holds across every pair."""
    if not pairs:
        return False, 1.0, 1.0
    fh, fw = None, None
    for i, o in pairs:
        ih, iw = i.shape
        oh, ow = o.shape
        if ih == 0 or iw == 0 or oh % ih != 0 or ow % iw != 0:
            return False, 1.0, 1.0
        sh, sw = oh // ih, ow // iw
        if fh is None:
            fh, fw = sh, sw
        elif (sh, sw) != (fh, fw):
            return False, 1.0, 1.0
        # Verify it is actually nearest-neighbor replication (not some
        # self-similar / pattern-aware scaling).
        if not np.array_equal(np.repeat(np.repeat(i, sh, axis=0), sw, axis=1), o):
            return False, 1.0, 1.0
    return True, float(fh or 1), float(fw or 1)


# ---- Local-rule neighborhood detection ---------------------------------------

def _local_rule_window_size(pairs, max_radius: int = 2) -> int:
    """Smallest kernel size k=2r+1 (r in 0..max_radius) such that, across all
    pairs, the output cell is a deterministic function of the input's kxk
    window centered on the same cell. Returns 99 if none of the tested radii
    suffice (or if shapes don't match).
    """
    if not pairs or any(i.shape != o.shape for i, o in pairs):
        return 99

    for r in range(max_radius + 1):
        lut: dict[tuple, int] = {}
        consistent = True
        for i, o in pairs:
            h, w = i.shape
            padded = np.pad(i, r, mode="constant", constant_values=-1)
            for y in range(h):
                for x in range(w):
                    window = tuple(padded[y:y + 2 * r + 1, x:x + 2 * r + 1].flatten().tolist())
                    out_val = int(o[y, x])
                    if window in lut:
                        if lut[window] != out_val:
                            consistent = False
                            break
                    else:
                        lut[window] = out_val
                if not consistent:
                    break
            if not consistent:
                break
        if consistent:
            return 2 * r + 1
    return 99


# ---- Misc helpers ------------------------------------------------------------

def _has_uniform_border(a: np.ndarray) -> bool:
    if a.size == 0 or a.shape[0] < 2 or a.shape[1] < 2:
        return False
    top, bot = a[0, :], a[-1, :]
    left, right = a[:, 0], a[:, -1]
    candidate = int(top[0])
    return all(np.all(edge == candidate) for edge in (top, bot, left, right))


def _background_color(pairs) -> int:
    counter: Counter[int] = Counter()
    for i, _ in pairs:
        counter.update(int(v) for v in i.flatten())
    return counter.most_common(1)[0][0] if counter else 0


# ---- Public API --------------------------------------------------------------

def extract_features(task: dict) -> dict[str, Any]:
    pairs = _train_arrays(task)
    all_pairs = _all_arrays(task)

    if not pairs:
        return {"_empty": True}

    shape_eq = _all(pairs, lambda i, o: i.shape == o.shape)
    in_square = _all(pairs, lambda i, _o: i.shape[0] == i.shape[1])
    out_square = _all(pairs, lambda _i, o: o.shape[0] == o.shape[1])
    transpose_shape = _all(pairs, lambda i, o: o.shape == (i.shape[1], i.shape[0]))

    scaled, sh, sw = _detect_scale(pairs)
    perm_map = _color_permutation(pairs)

    same_color_set = _all(pairs, lambda i, o: set(i.flatten().tolist()) == set(o.flatten().tolist()))
    color_count_preserved = _all(
        pairs, lambda i, o: Counter(i.flatten().tolist()) == Counter(o.flatten().tolist())
    )

    if shape_eq:
        diffs = [np.mean(i != o) for i, o in pairs]
        pixel_change_fraction = float(np.mean(diffs))
    else:
        pixel_change_fraction = 1.0

    out_colors: set[int] = set()
    for _, o in pairs:
        out_colors.update(int(v) for v in o.flatten())
    output_is_single_channel = len(out_colors) <= 2

    return {
        # Shape
        "output_shape_eq_input": shape_eq,
        "output_always_square": out_square,
        "input_always_square": in_square,
        "output_h_eq_input_w": transpose_shape,
        "output_is_input_scaled": scaled,
        "scale_factor_h": sh,
        "scale_factor_w": sw,

        # Color
        "same_color_set": same_color_set,
        "color_count_preserved": color_count_preserved,
        "is_color_permutation": perm_map is not None and shape_eq,
        "color_permutation_map": perm_map,
        "uses_only_two_colors": output_is_single_channel,
        "background_color": _background_color(pairs),

        # Geometric
        "is_identity": _is_identity(pairs),
        "is_rot90": _is_rot90(pairs),
        "is_rot180": _is_rot180(pairs),
        "is_rot270": _is_rot270(pairs),
        "is_flip_h": _is_flip_h(pairs),
        "is_flip_v": _is_flip_v(pairs),
        "is_transpose": _is_transpose(pairs),

        # Local rule
        "output_is_single_channel": output_is_single_channel,
        "pixel_change_fraction": pixel_change_fraction,
        "max_local_context_needed": _local_rule_window_size(all_pairs, max_radius=2),

        # Grid properties
        "input_has_border": all(_has_uniform_border(i) for i, _ in pairs),
        "output_has_border": all(_has_uniform_border(o) for _, o in pairs),
        "input_max_h": max(int(i.shape[0]) for i, _ in pairs),
        "input_max_w": max(int(i.shape[1]) for i, _ in pairs),
        "output_max_h": max(int(o.shape[0]) for _, o in pairs),
        "output_max_w": max(int(o.shape[1]) for _, o in pairs),
    }


__all__ = ["extract_features"]
