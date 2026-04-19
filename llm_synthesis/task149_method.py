def method(grid):
    import numpy as np
    out = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            r0 = i * 4
            c0 = j * 4
            block = grid[r0:r0+3, c0:c0+3]
            if np.sum(block == 6) >= 2:
                out[i, j] = 1
    return out
