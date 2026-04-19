def method(grid):
    import numpy as np
    g = np.asarray(grid)
    h, w = g.shape
    
    # Find separator rows and columns (all zeros)
    sep_rows = [i for i in range(h) if np.all(g[i] == 0)]
    sep_cols = [j for j in range(w) if np.all(g[:, j] == 0)]
    
    # Build row groups
    row_groups = []
    start = 0
    for r in sep_rows + [h]:
        if start < r:
            row_groups.append((start, r))
        start = r + 1
    
    col_groups = []
    start = 0
    for c in sep_cols + [w]:
        if start < c:
            col_groups.append((start, c))
        start = c + 1
    
    out = np.zeros((len(row_groups), len(col_groups)), dtype=g.dtype)
    for i, (r0, r1) in enumerate(row_groups):
        for j, (c0, c1) in enumerate(col_groups):
            block = g[r0:r1, c0:c1]
            vals = block[block != 0]
            if len(vals) > 0:
                out[i, j] = np.bincount(vals).argmax()
    return out
