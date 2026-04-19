import numpy as np

def method(grid):
    colors = []
    y0, y1 = 0, grid.shape[0] - 1
    x0, x1 = 0, grid.shape[1] - 1
    while True:
        c = grid[y0, x0]
        colors.append(int(c))
        sub = grid[y0:y1+1, x0:x1+1]
        mask = sub != c
        if not mask.any():
            break
        ys, xs = np.where(mask)
        ny0 = y0 + int(ys.min())
        ny1 = y0 + int(ys.max())
        nx0 = x0 + int(xs.min())
        nx1 = x0 + int(xs.max())
        # Safety: if bbox didn't shrink, stop
        if (ny0, ny1, nx0, nx1) == (y0, y1, x0, x1):
            break
        y0, y1, x0, x1 = ny0, ny1, nx0, nx1
    n = len(colors)
    size = 2 * n - 1
    out = np.zeros((size, size), dtype=grid.dtype)
    for i, c in enumerate(colors):
        out[i:size-i, i:size-i] = c
    return out
