def method(grid):
    out = np.zeros_like(grid)
    mid = grid.shape[1] // 2
    out[:, mid] = grid[:, mid]
    return out
