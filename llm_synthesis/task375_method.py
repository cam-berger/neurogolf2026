import numpy as np

def method(grid):
    out = grid.copy()
    pos = np.argwhere(grid == 0)
    r, c = pos[0]
    h, w = grid.shape
    for i in range(h):
        for j in range(w):
            if abs(i - r) == abs(j - c):
                out[i, j] = 0
    return out
