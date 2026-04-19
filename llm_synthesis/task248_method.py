def method(grid):
    import numpy as np
    h, w = grid.shape
    out = np.zeros_like(grid)
    r, c = h - 1, 0
    dr, dc = -1, 1
    for _ in range(h):
        out[r, c] = 1
        nr, nc = r + dr, c + dc
        if nc < 0 or nc >= w:
            dc = -dc
            nc = c + dc
        if nr < 0 or nr >= h:
            dr = -dr
            nr = r + dr
        r, c = nr, nc
    return out
