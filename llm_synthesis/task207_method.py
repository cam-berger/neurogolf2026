def method(grid):
    import numpy as np
    quads = [
        grid[0:2, 0:2],
        grid[0:2, 3:5],
        grid[3:5, 0:2],
        grid[3:5, 3:5],
    ]
    for i in range(4):
        if all(not np.array_equal(quads[i], quads[j]) for j in range(4) if j != i):
            return quads[i].copy()
    # fallback: the one that differs from majority
    for i in range(4):
        matches = sum(np.array_equal(quads[i], quads[j]) for j in range(4))
        if matches == 1:
            return quads[i].copy()
    return quads[0].copy()
