def method(grid):
    g = grid.copy()
    H, W = g.shape
    bottom = g[H-1]
    outer = bottom[0]
    inner_color = None
    for v in bottom:
        if v != outer and v != 0:
            inner_color = v
            break
    if inner_color is None:
        return g
    row2 = g[H-2]
    cols = np.where(row2 != 0)[0]
    if len(cols) == 0:
        return g
    left = cols[0]
    right = cols[-1]
    r, c = H-2, left
    while True:
        r -= 1
        c -= 1
        if r < 0 or c < 0:
            break
        g[r, c] = inner_color
    r, c = H-2, right
    while True:
        r -= 1
        c += 1
        if r < 0 or c >= W:
            break
        g[r, c] = inner_color
    return g
