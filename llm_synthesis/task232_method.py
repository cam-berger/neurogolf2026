import numpy as np

def method(grid):
    out = grid.copy()
    h, w = out.shape
    for r in range(h):
        for c in range(w):
            v = grid[r, c]
            if v != 0:
                for j in range(c+1, w):
                    out[r, j] = 5 if (j - c) % 2 == 1 else v
    return out
