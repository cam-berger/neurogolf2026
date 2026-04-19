def method(grid):
    g = np.array(grid)
    out = np.zeros_like(g)
    H, W = g.shape
    ys, xs = np.where(g != 0)
    if len(ys) == 0:
        return out
    r, c = ys[0], xs[0]
    color = g[r, c]
    for i in range(H):
        for j in range(W):
            if (i - j) == (r - c) or (i + j) == (r + c):
                out[i, j] = color
    return out
