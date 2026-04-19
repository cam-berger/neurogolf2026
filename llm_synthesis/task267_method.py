def method(grid):
    g = grid.copy()
    corner = g[-1, 0]
    g[-1, 0] = 0
    mask = g != 0
    g[mask] = corner
    return g
