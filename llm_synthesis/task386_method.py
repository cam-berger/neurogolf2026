def method(grid):
    import numpy as np
    grid = np.asarray(grid)
    # Find separator column of 1s
    sep = None
    for c in range(grid.shape[1]):
        if np.all(grid[:, c] == 1):
            sep = c
            break
    left = grid[:, :sep]
    right = grid[:, sep+1:]
    out = np.where((left == 0) & (right == 0), 3, 0)
    return out
