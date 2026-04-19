import numpy as np

def method(grid):
    out = grid.copy()
    pos = np.argwhere(grid == 8)
    if len(pos) == 0:
        return out
    r0, c0 = pos.min(axis=0)
    r1, c1 = pos.max(axis=0)
    sub = out[r0:r1+1, c0:c1+1]
    sub[sub == 1] = 3
    out[r0:r1+1, c0:c1+1] = sub
    return out
