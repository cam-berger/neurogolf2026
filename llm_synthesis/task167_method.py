def method(grid):
    import numpy as np
    n = len(np.unique(grid))
    out = np.zeros((3, 3), dtype=int)
    if n == 1:
        out[0, :] = 5
    elif n == 2:
        for i in range(3):
            out[i, i] = 5
    else:
        for i in range(3):
            out[i, 2 - i] = 5
    return out
