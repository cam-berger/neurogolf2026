def method(grid):
    import numpy as np
    scale = int(np.count_nonzero(grid))
    return np.kron(grid, np.ones((scale, scale), dtype=grid.dtype))
