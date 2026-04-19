import numpy as np

def method(grid):
    g = grid.copy()
    h, w = g.shape
    for c in range(w):
        rows = [r for r in range(h) if g[r, c] == 2]
        if not rows:
            continue
        n = len(rows)
        k = n // 2
        for r in rows[n - k:]:
            g[r, c] = 8
    return g
