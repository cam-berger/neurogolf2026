def method(grid):
    import numpy as np
    g = grid.copy()
    pos2 = tuple(np.argwhere(g == 2)[0])
    pos3 = tuple(np.argwhere(g == 3)[0])
    r2, c2 = pos2
    r3, c3 = pos3
    # Corner at (row of 2, col of 3)
    cr, cc = r2, c3
    # Horizontal line in row r2 between c2 and cc
    c_lo, c_hi = sorted([c2, cc])
    for c in range(c_lo, c_hi + 1):
        if g[r2, c] == 0:
            g[r2, c] = 8
    # Vertical line in col c3 between r3 and cr
    r_lo, r_hi = sorted([r3, cr])
    for r in range(r_lo, r_hi + 1):
        if g[r, c3] == 0:
            g[r, c3] = 8
    return g
