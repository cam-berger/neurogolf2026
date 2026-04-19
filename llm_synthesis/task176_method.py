def method(grid):
    import numpy as np
    g = grid.copy()
    H, W = g.shape
    k = 0
    for c in range(0, W + 3, 6):
        if k % 2 == 0:
            cells = [(1, c), (2, c - 1), (2, c), (2, c + 1)]
        else:
            cells = [(0, c - 1), (0, c), (0, c + 1), (1, c)]
        for r, cc in cells:
            if 0 <= r < H and 0 <= cc < W:
                if g[r, cc] == 0:
                    g[r, cc] = 4
        k += 1
    return g
