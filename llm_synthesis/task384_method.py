def method(grid):
    import numpy as np
    rows = np.any(grid != 0, axis=1)
    cols = np.any(grid != 0, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    cropped = grid[r0:r1+1, c0:c1+1]
    return np.kron(cropped, np.ones((2, 2), dtype=grid.dtype))
