def method(grid):
    import numpy as np
    grid = np.asarray(grid)
    H, W = grid.shape
    
    # Find separator rows (full rows of one color)
    sep_rows = [i for i in range(H) if len(np.unique(grid[i])) == 1]
    sep_cols = [j for j in range(W) if len(np.unique(grid[:, j])) == 1]
    
    # Determine separator color
    sep_color = grid[sep_rows[0], 0] if sep_rows else grid[0, sep_cols[0]]
    
    # Get row boundaries (excluding separators)
    row_starts = []
    row_ends = []
    in_block = False
    for i in range(H):
        if i not in sep_rows:
            if not in_block:
                row_starts.append(i)
                in_block = True
        else:
            if in_block:
                row_ends.append(i)
                in_block = False
    if in_block:
        row_ends.append(H)
    
    col_starts = []
    col_ends = []
    in_block = False
    for j in range(W):
        if j not in sep_cols:
            if not in_block:
                col_starts.append(j)
                in_block = True
        else:
            if in_block:
                col_ends.append(j)
                in_block = False
    if in_block:
        col_ends.append(W)
    
    n_rows = len(row_starts)
    n_cols = len(col_starts)
    
    compressed = np.zeros((n_rows, n_cols), dtype=grid.dtype)
    for r in range(n_rows):
        for c in range(n_cols):
            cell = grid[row_starts[r]:row_ends[r], col_starts[c]:col_ends[c]]
            vals = np.unique(cell)
            nonzero = vals[vals != 0]
            compressed[r, c] = nonzero[0] if len(nonzero) > 0 else 0
    
    return np.fliplr(compressed)
