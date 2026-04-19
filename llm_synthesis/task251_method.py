import numpy as np

def method(grid):
    g = grid.copy()
    H, W = g.shape
    for r1 in range(H):
        for r2 in range(r1 + 2, H):
            for c1 in range(W):
                for c2 in range(c1 + 2, W):
                    if (np.all(g[r1, c1:c2+1] == 2) and
                        np.all(g[r2, c1:c2+1] == 2) and
                        np.all(g[r1:r2+1, c1] == 2) and
                        np.all(g[r1:r2+1, c2] == 2)):
                        interior = g[r1+1:r2, c1+1:c2]
                        interior[interior == 0] = 1
    return g
