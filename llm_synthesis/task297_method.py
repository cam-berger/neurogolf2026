import numpy as np

def method(grid):
    g = grid.copy()
    h, w = g.shape
    colors = g[0]
    for r in range(2, h):
        g[r, :] = colors[(r - 2) % w]
    return g
