import numpy as np

def method(grid):
    g = grid.copy()
    # find bounding box of 5s
    rows, cols = np.where(g == 5)
    r0, r1 = rows.min(), rows.max()
    c0, c1 = cols.min(), cols.max()
    # fill interior with 8
    g[r0+1:r1, c0+1:c1] = 8
    # find gap on the border
    # top
    for c in range(c0, c1+1):
        if g[r0, c] == 0:
            for r in range(r0, -1, -1):
                g[r, c] = 8
    # bottom
    for c in range(c0, c1+1):
        if g[r1, c] == 0:
            for r in range(r1, g.shape[0]):
                g[r, c] = 8
    # left
    for r in range(r0, r1+1):
        if g[r, c0] == 0:
            for c in range(c0, -1, -1):
                g[r, c] = 8
    # right
    for r in range(r0, r1+1):
        if g[r, c1] == 0:
            for c in range(c1, g.shape[1]):
                g[r, c] = 8
    return g
