def method(grid):
    import numpy as np
    out = np.zeros_like(grid)
    colors = np.unique(grid)
    for c in colors:
        if c == 0:
            continue
        pts = np.argwhere(grid == c)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        out[r0:r1+1, c0:c1+1] = c
    return out
