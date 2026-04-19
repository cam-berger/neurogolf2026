import numpy as np

def method(grid):
    out = np.zeros((3, 3), dtype=grid.dtype)
    for i in range(3):
        for j in range(3):
            block = grid[i*3:(i+1)*3, j*3:(j+1)*3]
            vals = [v for v in block.flatten() if v != 0 and v != 5]
            if vals:
                out[i, j] = max(set(vals), key=vals.count)
    return out
