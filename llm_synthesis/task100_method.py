import numpy as np

def method(grid):
    g = np.asarray(grid)
    best_color = None
    best_area = -1
    for color in np.unique(g):
        if color == 0:
            continue
        mask = (g == color)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            continue
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area > best_area:
            best_area = area
            best_color = color
    return np.full((2, 2), best_color, dtype=g.dtype)
