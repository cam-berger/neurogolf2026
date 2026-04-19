import numpy as np

def method(grid):
    for color in np.unique(grid):
        if color == 0:
            continue
        rows, cols = np.where(grid == color)
        r0, r1 = rows.min(), rows.max()
        c0, c1 = cols.min(), cols.max()
        sub = grid[r0:r1+1, c0:c1+1]
        if np.any(sub == 0):
            return np.array([[color]])
    return np.array([[0]])
