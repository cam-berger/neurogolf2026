def method(grid):
    import numpy as np
    from collections import defaultdict
    
    g = grid.copy()
    H, W = g.shape
    
    # Find separator rows/cols (all same non-zero value)
    sep_rows = [r for r in range(H) if len(np.unique(g[r])) == 1 and g[r,0] != 0]
    sep_cols = [c for c in range(W) if len(np.unique(g[:,c])) == 1 and g[0,c] != 0]
    
    sep_color = g[sep_rows[0], 0]
    
    # Cell boundaries
    row_bounds = [0] + [r+1 for r in sep_rows]
    row_ends = sep_rows + [H]
    col_bounds = [0] + [c+1 for c in sep_cols]
    col_ends = sep_cols + [W]
    
    nR = len(row_bounds)
    nC = len(col_bounds)
    
    # Find color in each cell
    colors = {}
    for i in range(nR):
        for j in range(nC):
            r0, r1 = row_bounds[i], row_ends[i]
            c0, c1 = col_bounds[j], col_ends[j]
            block = g[r0:r1, c0:c1]
            vals = [v for v in block.flatten() if v != 0 and v != sep_color]
            if vals:
                colors[(i,j)] = vals[0]
    
    # Group cells by color
    by_color = defaultdict(list)
    for pos, col in colors.items():
        by_color[col].append(pos)
    
    # For each pair of same-colored cells in same row or column, fill between
    for col, positions in by_color.items():
        for a_idx in range(len(positions)):
            for b_idx in range(a_idx+1, len(positions)):
                a = positions[a_idx]
                b = positions[b_idx]
                if a[0] == b[0]:  # same row
                    j1, j2 = sorted([a[1], b[1]])
                    i = a[0]
                    for jj in range(j1, j2+1):
                        r0, r1 = row_bounds[i], row_ends[i]
                        c0, c1 = col_bounds[jj], col_ends[jj]
                        g[r0:r1, c0:c1] = col
                elif a[1] == b[1]:  # same col
                    i1, i2 = sorted([a[0], b[0]])
                    j = a[1]
                    for ii in range(i1, i2+1):
                        r0, r1 = row_bounds[ii], row_ends[ii]
                        c0, c1 = col_bounds[j], col_ends[j]
                        g[r0:r1, c0:c1] = col
    
    return g
