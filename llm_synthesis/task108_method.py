import numpy as np

def method(grid):
    h, w = grid.shape
    out = np.zeros((h*2, w*2), dtype=grid.dtype)
    for r in range(h):
        for c in range(w):
            v = grid[r, c]
            if v != 0:
                br, bc = (r-1)*2, (c-1)*2
                out[br:br+4, bc:bc+4] = v
    return out
