def method(grid):
    g = np.array(grid)
    H, W = g.shape
    
    # Find separator rows and columns (all 4s)
    sep_rows = [r for r in range(H) if np.all(g[r, :] == 4)]
    sep_cols = [c for c in range(W) if np.all(g[:, c] == 4)]
    
    # Build row and column block boundaries
    row_bounds = [-1] + sep_rows + [H]
    col_bounds = [-1] + sep_cols + [W]
    
    row_blocks = []
    for i in range(len(row_bounds) - 1):
        s, e = row_bounds[i] + 1, row_bounds[i+1]
        if s < e:
            row_blocks.append((s, e))
    
    col_blocks = []
    for i in range(len(col_bounds) - 1):
        s, e = col_bounds[i] + 1, col_bounds[i+1]
        if s < e:
            col_blocks.append((s, e))
    
    # Find source block (non-empty)
    source = None
    for (rs, re) in row_blocks:
        for (cs, ce) in col_blocks:
            block = g[rs:re, cs:ce]
            if np.any(block != 0):
                source = block.copy()
                break
        if source is not None:
            break
    
    out = g.copy()
    for (rs, re) in row_blocks:
        for (cs, ce) in col_blocks:
            if (re - rs, ce - cs) == source.shape:
                out[rs:re, cs:ce] = source
    
    return out
