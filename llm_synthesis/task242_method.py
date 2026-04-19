def method(grid):
    g = grid.copy()
    n = g.shape[0]
    zeros = np.argwhere(g == 0)
    r0, c0 = zeros.min(axis=0)
    out = np.zeros((3,3), dtype=g.dtype)
    for dr in range(3):
        for dc in range(3):
            r, c = r0+dr, c0+dc
            for rr, cc in [(n-1-r, c), (r, n-1-c), (n-1-r, n-1-c)]:
                if g[rr, cc] != 0:
                    out[dr, dc] = g[rr, cc]
                    break
    return out
