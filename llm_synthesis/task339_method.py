def method(grid):
    import numpy as np
    vals = grid[grid != 0]
    color = vals[0]
    return np.full((1, len(vals)), color, dtype=grid.dtype)
