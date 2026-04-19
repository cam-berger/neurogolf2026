import numpy as np

def method(grid):
    g = np.asarray(grid)
    rows, cols = g.shape
    # Find divider row and column (lines made of a single color)
    div_row = None
    div_col = None
    for r in range(rows):
        if len(set(g[r, :])) == 1:
            div_row = r
            break
    for c in range(cols):
        if len(set(g[:, c])) == 1:
            div_col = c
            break
    quadrants = [
        g[:div_row, :div_col],
        g[:div_row, div_col+1:],
        g[div_row+1:, :div_col],
        g[div_row+1:, div_col+1:],
    ]
    # Background color = most common
    vals, counts = np.unique(g, return_counts=True)
    # exclude divider color
    divider_color = g[div_row, 0]
    bg = None
    best = -1
    for v, c in zip(vals, counts):
        if v == divider_color:
            continue
        if c > best:
            best = c
            bg = v
    # Find quadrant with non-bg cell
    for q in quadrants:
        if np.any(q != bg):
            return q
    return quadrants[0]
