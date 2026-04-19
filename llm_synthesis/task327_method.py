def method(grid):
    n = grid.shape[0]
    size = 2 * n
    out = np.zeros((size, size), dtype=grid.dtype)
    for k in range(size):
        for i in range(n):
            for j in range(n):
                r, c = i + k, j + k
                if r < size and c < size and grid[i, j] != 0:
                    out[r, c] = grid[i, j]
    return out
