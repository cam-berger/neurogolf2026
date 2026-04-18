"""High-level helpers for defining local rules as plain Python functions.

Instead of writing ONNX plumbing, define a rule as a function that takes a
small grid neighborhood and returns the output color. The framework runs it
over every observed window, builds a LUT, compiles it to ONNX, and verifies
correctness — all in one call.

Usage (in custom_rules/task015.py):

    from custom_rules.helpers import make_rule

    def rule(w):
        # w is a Window — a 3x3 view with named accessors
        if w.center == 0 and w.count(1) >= 3:
            return 1
        return w.center

    generate = make_rule(15, rule, kernel=3)

That's it. `generate` is the function the registry expects.

For quick iteration from a REPL / notebook:

    from custom_rules.helpers import test_rule
    test_rule(15, rule, kernel=3)      # prints correctness + cost
    test_rule(15, rule, kernel=3, show_failures=True)  # also prints wrong cells
"""

from __future__ import annotations

import numpy as np
import onnx

from custom_rules.lut import compile_lut_to_onnx, extract_lut
from pipeline.loader import encode_grid, get_all_pairs, load_task


# ---------------------------------------------------------------------------
# Window: friendly view into a k×k neighborhood
# ---------------------------------------------------------------------------

class Window:
    """Lightweight view into a k×k neighborhood extracted from an ARC grid.

    Values are ints 0–9 for grid colors, or -1 for cells outside the grid.
    Named accessors assume a 3×3 kernel (center, top, left, …); for 5×5
    use w[row, col] indexing directly.
    """

    __slots__ = ("_data", "_k")

    def __init__(self, flat: tuple[int, ...], kernel: int):
        self._data = flat
        self._k = kernel

    # -- positional access --------------------------------------------------

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            r, c = idx
            return self._data[r * self._k + c]
        return self._data[idx]

    @property
    def flat(self) -> tuple[int, ...]:
        return self._data

    @property
    def array(self) -> np.ndarray:
        return np.array(self._data, dtype=np.int64).reshape(self._k, self._k)

    # -- 3×3 named accessors (center = k//2, k//2) -------------------------

    @property
    def center(self) -> int:
        m = self._k // 2
        return self._data[m * self._k + m]

    @property
    def top(self) -> int:
        m = self._k // 2
        return self._data[(m - 1) * self._k + m]

    @property
    def bottom(self) -> int:
        m = self._k // 2
        return self._data[(m + 1) * self._k + m]

    @property
    def left(self) -> int:
        m = self._k // 2
        return self._data[m * self._k + (m - 1)]

    @property
    def right(self) -> int:
        m = self._k // 2
        return self._data[m * self._k + (m + 1)]

    @property
    def top_left(self) -> int:
        m = self._k // 2
        return self._data[(m - 1) * self._k + (m - 1)]

    @property
    def top_right(self) -> int:
        m = self._k // 2
        return self._data[(m - 1) * self._k + (m + 1)]

    @property
    def bottom_left(self) -> int:
        m = self._k // 2
        return self._data[(m + 1) * self._k + (m - 1)]

    @property
    def bottom_right(self) -> int:
        m = self._k // 2
        return self._data[(m + 1) * self._k + (m + 1)]

    # -- cardinal / diagonal neighbor lists ---------------------------------

    @property
    def cardinal(self) -> list[int]:
        """[top, right, bottom, left]"""
        return [self.top, self.right, self.bottom, self.left]

    @property
    def diagonal(self) -> list[int]:
        """[top_left, top_right, bottom_right, bottom_left]"""
        return [self.top_left, self.top_right, self.bottom_right, self.bottom_left]

    @property
    def neighbors(self) -> list[int]:
        """All 8 surrounding cells (cardinal + diagonal)."""
        return self.cardinal + self.diagonal

    # -- utility methods ----------------------------------------------------

    def count(self, color: int) -> int:
        """How many cells in the full window have this color."""
        return sum(1 for v in self._data if v == color)

    def neighbor_count(self, color: int) -> int:
        """How many of the 8 neighbors have this color."""
        return sum(1 for v in self.neighbors if v == color)

    def cardinal_count(self, color: int) -> int:
        return sum(1 for v in self.cardinal if v == color)

    @property
    def colors(self) -> set[int]:
        """Set of colors present (excluding -1 = outside)."""
        return {v for v in self._data if v >= 0}

    @property
    def is_border(self) -> bool:
        """True if any cell in the window is outside the grid (-1)."""
        return -1 in self._data

    def __repr__(self) -> str:
        rows = []
        for r in range(self._k):
            row = self._data[r * self._k:(r + 1) * self._k]
            rows.append(" ".join("." if v == -1 else str(v) for v in row))
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Rule → LUT → ONNX pipeline
# ---------------------------------------------------------------------------

def _apply_rule(rule_fn, lut: dict[tuple, int], kernel: int) -> dict[tuple, int]:
    """Run the user rule function over every observed window in the LUT."""
    new_lut: dict[tuple, int] = {}
    for window_flat, expected_color in lut.items():
        w = Window(window_flat, kernel)
        predicted = rule_fn(w)
        if predicted is None:
            predicted = expected_color  # None = "keep existing" shorthand
        new_lut[window_flat] = int(predicted)
    return new_lut


def make_rule(task_id: int, rule_fn, kernel: int = 3):
    """Return a `generate(task, features)` function suitable for the registry.

    The returned function:
    1. Extracts the observed LUT from the task's full pool.
    2. Runs `rule_fn(Window)` on every observed window.
    3. Compiles the resulting LUT into an analytical ONNX network.
    """

    def generate(task: dict, features: dict) -> onnx.ModelProto | None:
        lut = extract_lut(task, kernel)
        if lut is None:
            return None
        new_lut = _apply_rule(rule_fn, lut, kernel)
        return compile_lut_to_onnx(new_lut, kernel)

    return generate


# ---------------------------------------------------------------------------
# Interactive testing (for REPL / notebook)
# ---------------------------------------------------------------------------

def _plot_window(ax, flat, kernel, title, colors):
    """Render a single k×k window as a colored grid on a matplotlib Axes."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    arc = [
        (0, 0, 0), (30/255, 147/255, 1), (250/255, 61/255, 49/255),
        (78/255, 204/255, 48/255), (1, 221/255, 0), (153/255,)*3,
        (229/255, 59/255, 163/255), (1, 133/255, 28/255),
        (136/255, 216/255, 241/255), (147/255, 17/255, 49/255),
        (0.85, 0.85, 0.85),  # 10 = outside (-1)
    ]
    cmap = ListedColormap(arc)
    norm = BoundaryNorm([-0.5 + i for i in range(12)], cmap.N)

    arr = list(flat)
    # Map -1 → 10 for display
    arr = [10 if v == -1 else v for v in arr]
    grid = np.array(arr, dtype=int).reshape(kernel, kernel)
    ax.imshow(grid, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    for y in range(kernel):
        for x in range(kernel):
            v = flat[y * kernel + x]
            label = "." if v == -1 else str(v)
            c = "white" if v in (0, 9, -1) else "black"
            ax.text(x, y, label, ha="center", va="center", fontsize=8, color=c)


def show_failures(task_id: int, rule_fn, kernel: int = 3, max_show: int = 20):
    """Plot mismatched windows as colored grids: input window, predicted, expected."""
    import matplotlib.pyplot as plt

    task = load_task(task_id)
    lut = extract_lut(task, kernel)
    if lut is None:
        print("No consistent LUT"); return

    new_lut = _apply_rule(rule_fn, lut, kernel)
    mismatches = [(w, new_lut[w], lut[w]) for w in lut if new_lut[w] != lut[w]]

    if not mismatches:
        print(f"No mismatches — rule is correct on all {len(lut)} windows!")
        return

    n = min(len(mismatches), max_show)
    fig, axes = plt.subplots(n, 1, figsize=(3.5, 1.6 * n),
                              gridspec_kw={"hspace": 0.6})
    if n == 1:
        axes = [axes]

    for i, (win_flat, pred, exp) in enumerate(mismatches[:n]):
        ax = axes[i]
        _plot_window(ax, win_flat, kernel,
                     f"window {i}: yours={pred}, expected={exp}", None)

    fig.suptitle(f"Task {task_id} — {len(mismatches)} mismatches "
                 f"({n} shown)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.show()


def test_rule(task_id: int, rule_fn=None, kernel: int = 3, plot_failures: bool = False):
    """Quick end-to-end test of a rule function against all pairs.

    If rule_fn is None, rubber-stamps the observed LUT (baseline check).
    Pass plot_failures=True to get a colored grid plot of every mismatch.
    """
    task = load_task(task_id)
    lut = extract_lut(task, kernel)
    if lut is None:
        print(f"Task {task_id}: no consistent {kernel}×{kernel} LUT")
        return

    if rule_fn is not None:
        new_lut = _apply_rule(rule_fn, lut, kernel)
        mismatches = {w: (new_lut[w], lut[w]) for w in lut if new_lut[w] != lut[w]}
        if mismatches:
            print(f"Rule disagrees with observed data on {len(mismatches)}/{len(lut)} windows")
            if plot_failures:
                show_failures(task_id, rule_fn, kernel)
            return
        compile_from = new_lut
    else:
        compile_from = lut

    model = compile_lut_to_onnx(compile_from, kernel)
    path = f"/tmp/_test_rule_{task_id:03d}.onnx"
    onnx.save(model, path)

    from pipeline.validator import check_correctness, compute_cost, validate_constraints
    c = check_correctness(path, task)
    v = validate_constraints(path)
    cost = compute_cost(path) if v["valid"] else {"valid": False}

    print(f"Task {task_id}:")
    print(f"  Correct: {c['n_correct']}/{c['n_pairs']}")
    print(f"  Valid: {v['valid']}  FileSize: {v.get('file_size', 0) / 1024:.1f}KB")
    print(f"  LUT size: {len(compile_from)} windows")
    if cost.get("valid"):
        print(f"  Score: {cost['score']:.2f}  (params={cost['n_params']} macs={cost['mac_ops']})")
    if v.get("violations"):
        print(f"  Violations: {v['violations']}")


def show_lut_sample(task_id: int, kernel: int = 3, n: int = 20,
                    color_filter: int | None = None):
    """Print a sample of observed LUT windows for manual inspection.

    Args:
        color_filter: if set, only show windows whose output is this color.
    """
    task = load_task(task_id)
    lut = extract_lut(task, kernel)
    if lut is None:
        print("No consistent LUT"); return

    items = list(lut.items())
    if color_filter is not None:
        items = [(w, c) for w, c in items if c == color_filter]

    for i, (window, color) in enumerate(items[:n]):
        w = Window(window, kernel)
        print(f"--- window {i} → color {color}")
        print(w)
        print()

    remaining = len(items) - n
    if remaining > 0:
        print(f"... {remaining} more (pass n= to see more, color_filter= to filter)")


__all__ = ["Window", "make_rule", "show_failures", "show_lut_sample", "test_rule"]
