import numpy as np

def method(grid):
    h, w = grid.shape
    vals, counts = np.unique(grid, return_counts=True)
    key = vals[np.argmax(counts)]
    out = np.zeros((h*h, w*w), dtype=grid.dtype)
    for i in range(h):
        for j in range(w):
            if grid[i, j] == key:
                out[i*h:(i+1)*h, j*w:(j+1)*w] = grid
    return out
