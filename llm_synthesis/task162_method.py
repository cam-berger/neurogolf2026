import numpy as np

def method(grid):
    out = grid.copy()
    work = grid.copy()
    h, w = grid.shape
    for r in range(h - 2):
        for c in range(w - 2):
            if np.all(work[r:r+3, c:c+3] == 0):
                out[r:r+3, c:c+3] = 1
                work[r:r+3, c:c+3] = -1
    return out
