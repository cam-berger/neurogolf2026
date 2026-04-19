def method(grid):
    import numpy as np
    out = np.zeros_like(grid)
    H, W = grid.shape
    
    # Find top row of color-1 object
    ones_rows = np.where(grid == 1)[0]
    if len(ones_rows) == 0:
        return grid.copy()
    top_one = ones_rows.min()
    
    # For each non-zero color, find its top row and shift
    colors = np.unique(grid)
    colors = [c for c in colors if c != 0]
    
    for c in colors:
        rs, cs = np.where(grid == c)
        top_c = rs.min()
        delta = top_one - top_c
        for r, col in zip(rs, cs):
            nr = r + delta
            if 0 <= nr < H:
                out[nr, col] = c
    
    return out
