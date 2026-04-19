def method(grid):
    import numpy as np
    g = np.asarray(grid)
    # bounding box of nonzero
    nz = np.argwhere(g != 0)
    r0, c0 = nz.min(0)
    r1, c1 = nz.max(0) + 1
    sub = g[r0:r1, c0:c1]
    # dedupe adjacent identical rows
    keep_rows = [0]
    for i in range(1, sub.shape[0]):
        if not np.array_equal(sub[i], sub[i-1]):
            keep_rows.append(i)
    sub = sub[keep_rows, :]
    # dedupe adjacent identical columns
    keep_cols = [0]
    for j in range(1, sub.shape[1]):
        if not np.array_equal(sub[:, j], sub[:, j-1]):
            keep_cols.append(j)
    sub = sub[:, keep_cols]
    return sub
