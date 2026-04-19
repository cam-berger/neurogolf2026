def method(grid):
    import numpy as np
    rows = np.any(grid != 0, axis=1)
    cols = np.any(grid != 0, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    sub = grid[r0:r1+1, c0:c1+1]
    return np.tile(sub, (1, 2))
