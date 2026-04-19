import numpy as np

def method(grid):
    g = grid.copy()
    pts = list(zip(*np.where(g == 8)))
    used = set()
    for i, (r1, c1) in enumerate(pts):
        for j, (r2, c2) in enumerate(pts):
            if i >= j:
                continue
            if r1 == r2 and c1 != c2:
                a, b = sorted([c1, c2])
                for c in range(a+1, b):
                    if g[r1, c] == 0:
                        g[r1, c] = 3
            elif c1 == c2 and r1 != r2:
                a, b = sorted([r1, r2])
                for r in range(a+1, b):
                    if g[r, c1] == 0:
                        g[r, c1] = 3
    return g
