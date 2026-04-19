def method(grid):
    k = len(set(grid.flatten()) - {0})
    return np.kron(grid, np.ones((k, k), dtype=grid.dtype))
