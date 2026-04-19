import numpy as np

def method(grid):
    g = grid.copy()
    h, w = g.shape
    rows = [r for r in range(h) if np.all(g[r, :] == 8)]
    cols = [c for c in range(w) if np.all(g[:, c] == 8)]
    r1, r2 = rows
    c1, c2 = cols
    # middle column band between c1 and c2
    # top: rows < r1
    g[:r1, c1+1:c2] = 2
    # bottom: rows > r2
    g[r2+1:, c1+1:c2] = 1
    # middle row band: between r1 and r2
    g[r1+1:r2, :c1] = 4
    g[r1+1:r2, c2+1:] = 3
    g[r1+1:r2, c1+1:c2] = 6
    return g
