def method(grid):
    import numpy as np
    vals, counts = np.unique(grid[grid != 0], return_counts=True)
    order = np.argsort(-counts)
    sorted_vals = vals[order]
    return sorted_vals.reshape(-1, 1)
