import numpy as np

def method(grid):
    H, W = grid.shape
    new_W = 2 * W
    out = np.zeros((H, new_W), dtype=grid.dtype)
    for r in range(H):
        row = grid[r]
        if np.all(row == 0):
            continue
        # find smallest period
        period = W
        for p in range(1, W + 1):
            if all(row[i] == row[i % p] for i in range(W)):
                period = p
                break
        for i in range(new_W):
            out[r, i] = row[i % period]
    return out
