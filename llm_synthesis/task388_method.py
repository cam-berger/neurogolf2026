def method(grid):
    g = grid.copy()
    h, w = g.shape
    out = g.copy()
    for c in range(w):
        if np.any(g[:, c] != 0):
            col = out[:, c]
            col[col == 0] = 8
            out[:, c] = col
    return np.tile(out, (2, 2))
