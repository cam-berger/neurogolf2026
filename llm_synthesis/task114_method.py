def method(grid):
    import numpy as np
    H, W = grid.shape
    out = np.zeros((H+2, W+2), dtype=grid.dtype)
    # middle rows: input with first/last column duplicated
    for i in range(H):
        out[i+1, 1:W+1] = grid[i]
        out[i+1, 0] = grid[i, 0]
        out[i+1, W+1] = grid[i, -1]
    # top row: 0, input[0], 0
    out[0, 1:W+1] = grid[0]
    # bottom row: 0, input[-1], 0
    out[H+1, 1:W+1] = grid[-1]
    return out
