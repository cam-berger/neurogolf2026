def method(grid):
    import numpy as np
    N = grid.shape[0]
    colors = grid[0, :].tolist()
    if 0 in colors:
        unit_len = colors.index(0)
    else:
        unit_len = N
    unit = colors[:unit_len]
    M = 2 * N
    out = np.zeros((M, M), dtype=grid.dtype)
    for r in range(M):
        for c in range(M):
            idx = max(r, c)
            out[r, c] = unit[idx % unit_len]
    return out
