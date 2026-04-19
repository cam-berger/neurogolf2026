def method(grid):
    out = grid.copy()
    mask = out == 5
    rows = np.where(mask)[0]
    out[mask] = grid[rows, 0]
    return out
