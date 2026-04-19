import numpy as np

def method(grid):
    g = grid.copy()
    n, m = g.shape
    g[n-1, 1:] = 4
    for i in range(n-1):
        j = m - 1 - i
        if 1 <= j < m:
            g[i, j] = 2
    return g
