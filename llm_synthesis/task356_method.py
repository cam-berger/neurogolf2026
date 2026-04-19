def method(grid):
    import numpy as np
    out = grid.copy()
    pts = list(zip(*np.where(grid == 8)))
    for i, (r1, c1) in enumerate(pts):
        for r2, c2 in pts[i+1:]:
            if r1 == r2:
                out[r1, min(c1, c2):max(c1, c2) + 1] = 8
            elif c1 == c2:
                out[min(r1, r2):max(r1, r2) + 1, c1] = 8
    return out
