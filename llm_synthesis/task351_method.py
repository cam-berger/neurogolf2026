def method(grid):
    H, W = grid.shape
    mask = (grid == 3)
    rows, cols = np.where(mask)
    r1, r2 = rows.min(), rows.max()
    c1, c2 = cols.min(), cols.max()
    out = np.zeros((r2 - r1 + 1, c2 - c1 + 1), dtype=grid.dtype)
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            candidates = [(H - 1 - r, c), (r, W - 1 - c), (H - 1 - r, W - 1 - c)]
            for rr, cc in candidates:
                if grid[rr, cc] != 3:
                    out[r - r1, c - c1] = grid[rr, cc]
                    break
    return out
