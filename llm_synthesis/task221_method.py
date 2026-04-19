def method(grid):
    import numpy as np
    g = np.asarray(grid)
    count = int(np.sum(g != 0))
    tiles = 9 - count
    size = 3 * tiles
    out = np.zeros((size, size), dtype=g.dtype)
    for i in range(count):
        r, c = divmod(i, tiles)
        out[r*3:r*3+3, c*3:c*3+3] = g
    return out
