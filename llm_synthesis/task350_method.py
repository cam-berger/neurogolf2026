def method(grid):
    g = grid.copy()
    h, w = g.shape
    
    # Process each row
    for r in range(h):
        cols = np.where(grid[r] == 1)[0]
        if len(cols) >= 2:
            lo, hi = cols.min(), cols.max()
            for c in range(lo + 1, hi):
                if g[r, c] != 1:
                    g[r, c] = 8
    
    # Process each column
    for c in range(w):
        rows = np.where(grid[:, c] == 1)[0]
        if len(rows) >= 2:
            lo, hi = rows.min(), rows.max()
            for r in range(lo + 1, hi):
                if g[r, c] != 1:
                    g[r, c] = 8
    
    return g
