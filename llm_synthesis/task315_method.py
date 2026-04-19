def method(grid):
    import numpy as np
    h, w = grid.shape
    out = np.zeros((h*3, w*3), dtype=grid.dtype)
    for i in range(h):
        for j in range(w):
            if grid[i, j] == 2:
                out[i*h:(i+1)*h, j*w:(j+1)*w] = grid
    return out
