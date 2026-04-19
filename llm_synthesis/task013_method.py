import numpy as np

def method(grid):
    g = np.zeros_like(grid)
    H, W = grid.shape
    pts = [(r, c, int(grid[r, c])) for r, c in zip(*np.nonzero(grid))]
    
    if W >= H:
        # extend along column axis
        pts.sort(key=lambda x: x[1])
        (_, c1, v1), (_, c2, v2) = pts
        d = c2 - c1
        period = 2 * d
        for start, v in [(c1, v1), (c2, v2)]:
            p = start
            while p < W:
                g[:, p] = v
                p += period
    else:
        # extend along row axis
        pts.sort(key=lambda x: x[0])
        (r1, _, v1), (r2, _, v2) = pts
        d = r2 - r1
        period = 2 * d
        for start, v in [(r1, v1), (r2, v2)]:
            p = start
            while p < H:
                g[p, :] = v
                p += period
    return g
