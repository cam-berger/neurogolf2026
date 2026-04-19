def method(grid):
    import numpy as np
    g = grid.copy()
    H, W = g.shape
    nonzero_cols = np.any(g != 0, axis=0)
    if not nonzero_cols.any():
        return g
    L = int(np.max(np.where(nonzero_cols)[0])) + 1
    period = None
    for p in range(1, L):
        ok = True
        for i in range(p, L):
            if not np.array_equal(g[:, i], g[:, i-p]):
                ok = False
                break
        if ok:
            period = p
            break
    if period is None:
        return g
    for i in range(L, W):
        g[:, i] = g[:, i - period]
    return g
