def method(grid):
    import numpy as np
    g = np.asarray(grid)
    h, w = g.shape
    out = np.zeros_like(g)
    
    row_is_5 = np.all(g == 5, axis=1)
    col_is_5 = np.all(g == 5, axis=0)
    
    if row_is_5.any():
        bar_rows = np.where(row_is_5)[0]
        top = bar_rows.min()
        bot = bar_rows.max()
        out[top:bot+1, :] = 5
        for c in range(w):
            above = g[:top, c]
            below = g[bot+1:, c]
            above_count = int(np.sum((above != 0) & (above != 5)))
            below_count = int(np.sum((below != 0) & (below != 5)))
            for k in range(1, above_count+1):
                if top - k >= 0:
                    out[top-k, c] = 5
            for k in range(1, below_count+1):
                if bot + k < h:
                    out[bot+k, c] = 5
    else:
        bar_cols = np.where(col_is_5)[0]
        left = bar_cols.min()
        right = bar_cols.max()
        out[:, left:right+1] = 5
        for r in range(h):
            lseg = g[r, :left]
            rseg = g[r, right+1:]
            left_count = int(np.sum((lseg != 0) & (lseg != 5)))
            right_count = int(np.sum((rseg != 0) & (rseg != 5)))
            for k in range(1, left_count+1):
                if left - k >= 0:
                    out[r, left-k] = 5
            for k in range(1, right_count+1):
                if right + k < w:
                    out[r, right+k] = 5
    
    return out
