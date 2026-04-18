"""ASCII grid visualizer for debugging."""

from __future__ import annotations

_GLYPHS = "0123456789"


def render_grid(grid: list[list[int]]) -> str:
    """Render a grid as a string using digits 0-9, '.' = no color, '?' = overload."""
    rows = []
    for row in grid:
        chars = []
        for cell in row:
            if 0 <= cell < 10:
                chars.append(_GLYPHS[cell])
            elif cell == 10:
                chars.append(".")
            else:
                chars.append("?")
        rows.append("".join(chars))
    return "\n".join(rows)


def render_pair(input_grid, output_grid, label: str = "") -> str:
    sep = "  ->  "
    in_lines = render_grid(input_grid).splitlines() or [""]
    out_lines = render_grid(output_grid).splitlines() or [""]
    height = max(len(in_lines), len(out_lines))
    in_w = max((len(line) for line in in_lines), default=0)
    in_lines += [" " * in_w] * (height - len(in_lines))
    out_lines += [""] * (height - len(out_lines))
    body = "\n".join(f"{a.ljust(in_w)}{sep if i == 0 else ' ' * len(sep)}{b}"
                     for i, (a, b) in enumerate(zip(in_lines, out_lines)))
    return f"{label}\n{body}" if label else body


__all__ = ["render_grid", "render_pair"]
