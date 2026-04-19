def method(grid):
    import numpy as np
    g = grid.copy()
    h, w = g.shape
    # count 5s in last column
    n = int(np.sum(g[:, -1] == 5))
    # find the non-zero, non-5 color
    colors = np.unique(g)
    color = [c for c in colors if c != 0 and c != 5][0]
    # find vertical column and horizontal row
    # horizontal row: row entirely filled with color
    hrow = None
    for r in range(h):
        if np.all(g[r, :] == color):
            hrow = r
            break
    # vertical col: col entirely filled with color
    vcol = None
    for c in range(w):
        if np.all(g[:, c] == color):
            vcol = c
            break
    # new positions
    new_vcol = vcol - n
    new_hrow = hrow + n
    out = np.zeros_like(g)
    out[new_hrow, :] = color
    out[:, new_vcol] = color
    return out
