def method(grid):
    import numpy as np
    H, W = grid.shape
    out = np.full((H, W), 8, dtype=grid.dtype)
    r, c = H - 1, 0
    dr, dc = -1, 1
    for _ in range(H):
        out[r, c] = 1
        nr, nc = r + dr, c + dc
        if nc < 0 or nc >= W:
            dc = -dc
            nc = c + dc
        if nr < 0 or nr >= H:
            dr = -dr
            nr = r + dr
        r, c = nr, nc
    return out
