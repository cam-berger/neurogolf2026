import numpy as np

def method(grid):
    g = grid.copy()
    rows, cols = np.where(g == 8)
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            v = grid[r, c]
            if v == 0 or v == 8:
                continue
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if r0 <= r <= r1:
                if c < c0:
                    g[r, c0] = v
                else:
                    g[r, c1] = v
            elif c0 <= c <= c1:
                if r < r0:
                    g[r0, c] = v
                else:
                    g[r1, c] = v
    return g
