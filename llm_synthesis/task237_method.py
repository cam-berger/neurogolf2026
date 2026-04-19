def method(grid):
    g = grid.copy()
    h, w = g.shape
    cells = []
    for r in range(h):
        for c in range(w):
            if g[r, c] != 0:
                cells.append((r, c, g[r, c]))
    cells.sort()
    for i, (r, c, v) in enumerate(cells):
        g[r, c:w] = v
        next_r = cells[i+1][0] if i+1 < len(cells) else h
        for rr in range(r+1, next_r):
            g[rr, w-1] = v
    return g
