def method(grid):
    import numpy as np
    h = grid.shape[0]
    period = 2 * (h - 1)
    n_out = 2 * period + 1
    out = np.zeros((n_out, grid.shape[1]), dtype=grid.dtype)
    for i in range(n_out):
        k = i % period
        if k >= h:
            k = period - k
        out[i] = grid[k]
    return out
