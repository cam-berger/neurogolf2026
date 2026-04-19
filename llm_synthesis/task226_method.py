def method(grid):
    import numpy as np
    g = grid.copy()
    h, w = g.shape
    
    # Find rows/cols that are entirely 5
    full_rows = [i for i in range(h) if np.all(g[i, :] == 5)]
    full_cols = [j for j in range(w) if np.all(g[:, j] == 5)]
    
    # Build row cell ranges (contiguous non-full rows)
    row_ranges = []
    start = 0
    for r in full_rows:
        if start <= r - 1:
            row_ranges.append((start, r - 1))
        start = r + 1
    if start <= h - 1:
        row_ranges.append((start, h - 1))
    
    col_ranges = []
    start = 0
    for c in full_cols:
        if start <= c - 1:
            col_ranges.append((start, c - 1))
        start = c + 1
    if start <= w - 1:
        col_ranges.append((start, w - 1))
    
    nR = len(row_ranges)
    nC = len(col_ranges)
    
    def fill(ri, ci, val):
        r0, r1 = row_ranges[ri]
        c0, c1 = col_ranges[ci]
        g[r0:r1+1, c0:c1+1] = val
    
    # Top-left
    fill(0, 0, 1)
    # Middle
    fill(nR // 2, nC // 2, 2)
    # Bottom-right
    fill(nR - 1, nC - 1, 3)
    
    return g
