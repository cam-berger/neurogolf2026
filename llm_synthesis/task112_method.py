def method(grid):
    import numpy as np
    out = grid.copy()
    threes = np.argwhere(grid == 3)
    r0 = threes[:, 0].min()
    c0 = threes[:, 1].min()
    # 3-block is 2x2 starting at (r0, c0)
    twos = np.argwhere(grid == 2)
    H, W = grid.shape
    for r, c in twos:
        rr = 2 * r0 + 1 - r
        cc = 2 * c0 + 1 - c
        for nr, nc in [(r, c), (rr, c), (r, cc), (rr, cc)]:
            if 0 <= nr < H and 0 <= nc < W:
                out[nr, nc] = 2
    return out
