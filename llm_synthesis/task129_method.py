import numpy as np

def method(grid):
    vals, counts = np.unique(grid, return_counts=True)
    most = vals[np.argmax(counts)]
    return np.full_like(grid, most)
