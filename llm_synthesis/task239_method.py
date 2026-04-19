def method(grid):
    import numpy as np
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    vals = vals[order]
    counts = counts[order]
    H = int(counts[0])
    W = len(vals)
    out = np.zeros((H, W), dtype=grid.dtype)
    for i, (v, c) in enumerate(zip(vals, counts)):
        out[:c, i] = v
    return out
