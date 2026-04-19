def method(grid):
    import numpy as np
    g = np.array(grid)
    H, W = g.shape
    
    # Separator rows/cols: those with no zeros
    sep_rows = [r for r in range(H) if 0 not in g[r]]
    sep_cols = [c for c in range(W) if 0 not in g[:, c]]
    
    # Identify the separator color (most frequent in a sep row)
    if sep_rows:
        vals, counts = np.unique(g[sep_rows[0]], return_counts=True)
        sep_color = int(vals[np.argmax(counts)])
    else:
        sep_color = 0
    
    cells = []
    n_rows = len(sep_rows) - 1
    n_cols = len(sep_cols) - 1
    
    for i in range(n_rows):
        for j in range(n_cols):
            r1, r2 = sep_rows[i], sep_rows[i+1]
            c1, c2 = sep_cols[j], sep_cols[j+1]
            corners = [int(g[r1, c1]), int(g[r1, c2]),
                       int(g[r2, c1]), int(g[r2, c2])]
            if corners[0] != sep_color and corners[0] != 0 \
               and all(cc == corners[0] for cc in corners):
                cells.append((i, j, corners[0]))
    
    if not cells:
        return np.zeros((3, 3), dtype=int)
    
    min_r = min(c[0] for c in cells)
    max_r = max(c[0] for c in cells)
    min_c = min(c[1] for c in cells)
    max_c = max(c[1] for c in cells)
    
    out = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for i, j, col in cells:
        out[i - min_r, j - min_c] = col
    return out
