def method(grid):
    import numpy as np
    g = np.asarray(grid)
    H, W = g.shape
    out = np.zeros_like(g)
    
    # First, draw vertical lines for 2s
    cols_2 = set()
    for r in range(H):
        for c in range(W):
            if g[r, c] == 2:
                cols_2.add(c)
    for c in cols_2:
        out[:, c] = 2
    
    # Then, draw horizontal lines for 1s and 3s (overwriting)
    for r in range(H):
        for c in range(W):
            v = g[r, c]
            if v == 1 or v == 3:
                out[r, :] = v
    
    return out
