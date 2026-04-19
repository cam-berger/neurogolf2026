def method(grid):
    import numpy as np
    g = grid.copy()
    H, W = g.shape
    # find the 2-row
    rows = np.where((g == 2).any(axis=1))[0]
    r = rows[0]
    L = int((g[r] == 2).sum())
    out = np.zeros_like(g)
    out[r, :L] = 2
    # above: row r-k has length L+k filled with 3
    for k in range(1, r+1):
        length = min(L + k, W)
        out[r-k, :length] = 3
    # below: row r+k has length L-k filled with 1
    for k in range(1, L):
        if r + k >= H:
            break
        length = L - k
        out[r+k, :length] = 1
    return out
