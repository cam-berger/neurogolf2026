def method(grid):
    import numpy as np
    out = grid.copy()
    r8, c8 = [int(x[0]) for x in np.where(grid == 8)]
    r2, c2 = [int(x[0]) for x in np.where(grid == 2)]
    cr, cc = r2, c8
    # vertical between 8 and corner
    r1, r2v = sorted([r8, cr])
    for r in range(r1, r2v + 1):
        if out[r, c8] == 0:
            out[r, c8] = 4
    # horizontal between corner and 2
    c1, c2h = sorted([cc, c2])
    for c in range(c1, c2h + 1):
        if out[cr, c] == 0:
            out[cr, c] = 4
    return out
