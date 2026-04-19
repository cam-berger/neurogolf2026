def method(grid):
    g = grid.copy()
    ones = np.argwhere(g == 1)
    for r, c in ones:
        g[r, c] = 0
        g[r+2, c] = 1
    return g
