def method(grid):
    import numpy as np
    grid = np.asarray(grid)
    H, W = grid.shape
    for p in range(2, 11):
        tile = np.zeros((p, p), dtype=int)
        ok = True
        for r in range(H):
            if not ok:
                break
            for c in range(W):
                v = int(grid[r, c])
                if v == 0:
                    continue
                tr, tc = r % p, c % p
                if tile[tr, tc] == 0:
                    tile[tr, tc] = v
                elif tile[tr, tc] != v:
                    ok = False
                    break
        if ok and not (tile == 0).any():
            out = np.zeros_like(grid)
            for r in range(H):
                for c in range(W):
                    out[r, c] = tile[r % p, c % p]
            return out
    return grid
