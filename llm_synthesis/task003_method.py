def method(grid):
    import numpy as np
    g = grid.copy()
    n = g.shape[0]
    p = 1
    for cand in range(1, n):
        if np.array_equal(g[:n-cand], g[cand:]):
            p = cand
            break
    out = np.zeros((9, g.shape[1]), dtype=g.dtype)
    out[:n] = g
    for i in range(n, 9):
        out[i] = out[i-p]
    out[out == 1] = 2
    return out
