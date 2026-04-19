def method(grid):
    out = grid.copy()
    H, W = grid.shape
    used = np.zeros_like(grid, dtype=bool)
    for r in range(H - 1):
        for c in range(W - 1):
            if used[r, c]:
                continue
            block = grid[r:r+2, c:c+2]
            if np.all(block != 0):
                n = len(np.unique(block))
                used[r:r+2, c:c+2] = True
                for i in range(n):
                    rr = r + 2 + i
                    if rr < H:
                        out[rr, c:c+2] = 3
    return out
