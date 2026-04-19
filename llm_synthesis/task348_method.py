def method(grid):
    g = np.array(grid)
    out = g.copy()
    ys, xs = np.where(g == 7)
    H, W = g.shape
    
    if len(set(xs.tolist())) == 1:
        # vertical line
        c = xs[0]
        tip_row = ys.max()
        for r in range(tip_row + 1):
            k = tip_row - r
            for d in range(-k, k + 1):
                col = c + d
                if 0 <= col < W:
                    out[r, col] = 7 if d % 2 == 0 else 8
    else:
        # horizontal line
        r0 = ys[0]
        tip_col = xs.max()
        for c in range(tip_col + 1):
            k = tip_col - c
            for d in range(-k, k + 1):
                row = r0 + d
                if 0 <= row < H:
                    out[row, c] = 7 if d % 2 == 0 else 8
    return out
