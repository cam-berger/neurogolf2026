def method(grid):
    import numpy as np
    mask = grid != 1
    if not mask.any():
        return grid.copy()
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    sub = grid[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1].copy()
    sub[sub == 1] = 0
    return sub
