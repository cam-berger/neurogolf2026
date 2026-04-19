def method(grid):
    import numpy as np
    colors = np.unique(grid)
    mean_rows = {}
    mean_cols = {}
    for c in colors:
        ys, xs = np.where(grid == c)
        mean_rows[c] = ys.mean()
        mean_cols[c] = xs.mean()
    
    row_range = max(mean_rows.values()) - min(mean_rows.values())
    col_range = max(mean_cols.values()) - min(mean_cols.values())
    
    if col_range >= row_range:
        ordered = sorted(colors, key=lambda c: mean_cols[c])
        return np.array([ordered])
    else:
        ordered = sorted(colors, key=lambda c: mean_rows[c])
        return np.array([[c] for c in ordered])
