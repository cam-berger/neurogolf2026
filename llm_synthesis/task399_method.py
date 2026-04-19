def method(grid):
    import numpy as np
    n = int(np.sum(grid == 2) // 4)
    positions = [(0,0), (0,2), (1,1), (2,0), (2,2)]
    out = np.zeros((3,3), dtype=grid.dtype)
    for i in range(min(n, len(positions))):
        r, c = positions[i]
        out[r, c] = 1
    return out
