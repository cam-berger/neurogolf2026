def method(grid):
    g = np.array(grid)
    H, W = g.shape
    tl = g[0:2, 0:2]
    tr = g[0:2, W-2:W]
    bl = g[H-2:H, 0:2]
    br = g[H-2:H, W-2:W]
    out = np.zeros((3, 3), dtype=g.dtype)
    out[0:2, 0:2] = np.maximum(out[0:2, 0:2], tl)
    out[0:2, 1:3] = np.maximum(out[0:2, 1:3], tr)
    out[1:3, 0:2] = np.maximum(out[1:3, 0:2], bl)
    out[1:3, 1:3] = np.maximum(out[1:3, 1:3], br)
    return out
