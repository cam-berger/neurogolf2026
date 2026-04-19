def method(grid):
    import numpy as np
    H, W = grid.shape
    best_n = None
    for n in range(2, 10):
        ok = True
        for i in range(H):
            for j in range(W):
                v = grid[i, j]
                if v != 0:
                    if ((i + j) % n) + 1 != v:
                        ok = False
                        break
            if not ok:
                break
        if ok:
            best_n = n
            break
    out = np.zeros_like(grid)
    for i in range(H):
        for j in range(W):
            out[i, j] = ((i + j) % best_n) + 1
    return out
