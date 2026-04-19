def method(grid):
    import numpy as np
    g = np.asarray(grid)
    row = g[0]
    w = len(row)
    color = row[row != 0][0] if np.any(row != 0) else 0
    initial = int(np.sum(row != 0))
    h = w // 2
    out = np.zeros((h, w), dtype=g.dtype)
    for i in range(h):
        count = min(initial + i, w)
        out[i, :count] = color
    return out
