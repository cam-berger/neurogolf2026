import numpy as np

def method(grid):
    g = grid.copy()
    rows, cols = g.shape
    for r in range(rows):
        if g[r, 0] != 0 and g[r, 0] == g[r, -1]:
            g[r, :] = g[r, 0]
    return g
