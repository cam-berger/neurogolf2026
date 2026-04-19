def method(grid):
    g = grid.copy()
    h, w = g.shape
    cols = [(c, g[0, c]) for c in range(w) if g[0, c] != 0]
    for c, color in cols:
        for r in range(h):
            if r % 2 == 0:
                if 0 <= c < w:
                    g[r, c] = color
            else:
                if c - 1 >= 0:
                    g[r, c - 1] = color
                if c + 1 < w:
                    g[r, c + 1] = color
    return g
