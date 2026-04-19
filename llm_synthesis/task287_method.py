def method(grid):
    g = grid.copy()
    h, w = g.shape
    for i in range(h):
        for j in range(w):
            if g[i, j] == 4:
                for ii, jj in [(h-1-i, j), (i, w-1-j), (h-1-i, w-1-j)]:
                    if g[ii, jj] != 4:
                        g[i, j] = g[ii, jj]
                        break
    return g
