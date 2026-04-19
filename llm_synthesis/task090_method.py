def method(grid):
    g = grid.copy()
    H, W = g.shape
    best_area = 0
    best = None
    for r1 in range(H):
        for r2 in range(r1 + 1, H):  # at least 2 rows tall
            valid = np.all(g[r1:r2+1] == 0, axis=0)
            cur = 0
            for c in range(W):
                if valid[c]:
                    cur += 1
                    if cur >= 2:  # at least 2 cols wide
                        area = (r2 - r1 + 1) * cur
                        if area > best_area:
                            best_area = area
                            best = (r1, r2, c - cur + 1, c)
                else:
                    cur = 0
    if best is not None:
        r1, r2, c1, c2 = best
        g[r1:r2+1, c1:c2+1] = 6
    return g
