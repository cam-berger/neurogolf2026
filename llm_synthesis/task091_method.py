def method(grid):
    import numpy as np
    g = np.asarray(grid)
    fives = np.argwhere(g == 5)
    rows = fives[:, 0]
    cols = fives[:, 1]
    H, W = g.shape

    # Try vertical orientation: two columns of 5s capped by 8s above/below
    unique_cols = np.unique(cols)
    if len(unique_cols) == 2:
        c1, c2 = int(unique_cols[0]), int(unique_cols[1])
        rmin = int(rows.min())
        rmax = int(rows.max())
        if (rmin - 1 >= 0 and rmax + 1 < H and
            g[rmin - 1, c1] == 8 and g[rmax + 1, c1] == 8 and
            g[rmin - 1, c2] == 8 and g[rmax + 1, c2] == 8):
            return g[rmin - 1:rmax + 2, c1:c2 + 1].copy()

    # Try horizontal orientation: two rows of 5s capped by 8s left/right
    unique_rows = np.unique(rows)
    if len(unique_rows) == 2:
        r1, r2 = int(unique_rows[0]), int(unique_rows[1])
        cmin = int(cols.min())
        cmax = int(cols.max())
        if (cmin - 1 >= 0 and cmax + 1 < W and
            g[r1, cmin - 1] == 8 and g[r1, cmax + 1] == 8 and
            g[r2, cmin - 1] == 8 and g[r2, cmax + 1] == 8):
            return g[r1:r2 + 1, cmin - 1:cmax + 2].copy()

    # Fallback: just bounding box of 5s extended by 1
    if len(fives) > 0:
        rmin, rmax = int(rows.min()), int(rows.max())
        cmin, cmax = int(cols.min()), int(cols.max())
        r0 = max(0, rmin - 1); r1e = min(H, rmax + 2)
        c0 = max(0, cmin - 1); c1e = min(W, cmax + 2)
        return g[r0:r1e, c0:c1e].copy()
    return g.copy()
