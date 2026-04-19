def method(grid):
    g = grid.copy()
    h, w = g.shape
    clean_rows = []
    clean_cols = []
    for r in range(1, h-1):
        if np.all(g[r, 1:w-1] == 0):
            clean_rows.append(r)
    for c in range(1, w-1):
        if np.all(g[1:h-1, c] == 0):
            clean_cols.append(c)
    for r in clean_rows:
        mask = g[r] == 0
        g[r, mask] = 3
    for c in clean_cols:
        mask = g[:, c] == 0
        g[mask, c] = 3
    return g
