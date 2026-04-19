def method(grid):
    import numpy as np
    rows, cols = np.where(grid == 5)
    if len(rows) == 0:
        return np.zeros((9, 9), dtype=grid.dtype)
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    region = grid[r0:r1+1, c0:c1+1]
    h, w = region.shape
    bh = h // 3
    bw = w // 3
    P = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            if region[i*bh, j*bw] == 5:
                P[i, j] = 1
    out = np.kron(P, P) * 5
    return out.astype(grid.dtype)
