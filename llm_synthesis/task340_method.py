def method(grid):
    g = grid.copy()
    h, w = g.shape
    top = g[0, 1]
    bottom = g[h-1, 1]
    left = g[1, 0]
    right = g[1, w-1]
    
    out = np.zeros_like(g)
    out[0, :] = g[0, :]
    out[h-1, :] = g[h-1, :]
    out[:, 0] = g[:, 0]
    out[:, w-1] = g[:, w-1]
    
    for r in range(1, h-1):
        for c in range(1, w-1):
            v = g[r, c]
            if v == 0:
                continue
            if v == top:
                out[1, c] = v
            elif v == bottom:
                out[h-2, c] = v
            elif v == left:
                out[r, 1] = v
            elif v == right:
                out[r, w-2] = v
    return out
