import numpy as np

def method(grid):
    g = np.asarray(grid)
    h, w = g.shape
    out = np.zeros_like(g)
    for c in range(w):
        col = g[:, c]
        nz = col[col != 0]
        out[h - len(nz):, c] = nz
    return out
