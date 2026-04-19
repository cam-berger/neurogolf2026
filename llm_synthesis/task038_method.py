import numpy as np

def method(grid):
    h, w = grid.shape
    count = 0
    seen = set()
    for i in range(h - 1):
        for j in range(w - 1):
            if (grid[i, j] == 1 and grid[i+1, j] == 1 and 
                grid[i, j+1] == 1 and grid[i+1, j+1] == 1):
                if (i, j) not in seen:
                    count += 1
                    seen.update([(i, j), (i+1, j), (i, j+1), (i+1, j+1)])
    out = np.zeros((1, 5), dtype=int)
    out[0, :min(count, 5)] = 1
    return out
