def method(grid):
    import numpy as np
    nz = np.argwhere(grid != 0)
    r0, c0 = nz.min(axis=0)
    return grid[r0:r0+3, c0:c0+3].copy()
