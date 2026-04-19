def method(grid):
    import numpy as np
    g = np.array(grid)
    H, W = g.shape
    rows_with = np.where(np.any(g != 0, axis=1))[0]
    r0, r1 = rows_with.min(), rows_with.max()
    h = r1 - r0 + 1
    out = np.zeros_like(g)
    for r in range(H):
        src = r0 + (r - r0) % h
        out[r] = g[src]
    return out
