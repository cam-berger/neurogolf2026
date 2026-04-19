def method(grid):
    import numpy as np
    g = grid.copy()
    H, W = g.shape
    
    # Find divider color D (color that forms full rows/columns)
    D = None
    for r in range(H):
        if len(set(g[r])) == 1 and g[r, 0] != 0:
            D = int(g[r, 0])
            break
    
    # Find divider rows/columns
    div_rows = [r for r in range(H) if np.all(g[r] == D)]
    div_cols = [c for c in range(W) if np.all(g[:, c] == D)]
    
    row_bounds = [0] + [r + 1 for r in div_rows]
    row_ends = div_rows + [H]
    col_bounds = [0] + [c + 1 for c in div_cols]
    col_ends = div_cols + [W]
    
    nR = len(row_bounds)
    nC = len(col_bounds)
    
    # Find foreground color F (non-zero, non-D)
    F = None
    for v in np.unique(g):
        if v != 0 and v != D:
            F = int(v)
            break
    
    cell_h = row_ends[0] - row_bounds[0]
    cell_w = col_ends[0] - col_bounds[0]
    
    # Build union mask of all foreground positions (relative to cell)
    union = np.zeros((cell_h, cell_w), dtype=bool)
    for i in range(nR):
        for j in range(nC):
            sub = g[row_bounds[i]:row_ends[i], col_bounds[j]:col_ends[j]]
            if sub.shape == (cell_h, cell_w):
                union |= (sub == F)
    
    # Apply template to each cell
    out = g.copy()
    for i in range(nR):
        for j in range(nC):
            r0 = row_bounds[i]
            c0 = col_bounds[j]
            for rr in range(cell_h):
                for cc in range(cell_w):
                    if r0 + rr >= H or c0 + cc >= W:
                        continue
                    if union[rr, cc] and out[r0 + rr, c0 + cc] != F:
                        out[r0 + rr, c0 + cc] = D
    
    return out
