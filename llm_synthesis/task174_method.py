import numpy as np

def method(grid):
    grid = np.asarray(grid)
    colors = [c for c in np.unique(grid) if c != 0]
    for c in colors:
        mask = (grid == c)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        sub = grid[r0:r1+1, c0:c1+1].copy()
        sub[sub != c] = 0
        if np.array_equal(sub, sub[:, ::-1]):
            return sub
    return grid
