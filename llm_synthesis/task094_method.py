def method(grid):
    g = grid.copy()
    h, w = g.shape
    # find 5x5 boxes (top-left corners where we see a 1-bordered rectangle)
    centers = []
    for r in range(h - 4):
        for c in range(w - 4):
            if g[r, c] == 1 and g[r, c+4] == 1 and g[r+4, c] == 1 and g[r+4, c+4] == 1:
                # check it's a box
                if all(g[r, c+k] == 1 for k in range(5)) and all(g[r+4, c+k] == 1 for k in range(5)) \
                   and all(g[r+k, c] == 1 for k in range(5)) and all(g[r+k, c+4] == 1 for k in range(5)):
                    centers.append((r+2, c+2))
    for cr, cc in centers:
        for j in range(w):
            if g[cr, j] != 1:
                g[cr, j] = 6
        for i in range(h):
            if g[i, cc] != 1:
                g[i, cc] = 6
    return g
