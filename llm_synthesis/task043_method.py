import numpy as np

def method(grid):
    out = grid.copy()
    top_cols = [c for c in range(grid.shape[1]-1) if grid[0, c] == 5]
    for r in range(1, grid.shape[0]):
        if grid[r, -1] == 5:
            for c in top_cols:
                out[r, c] = 2
    return out
