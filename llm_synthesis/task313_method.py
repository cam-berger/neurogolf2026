def method(grid):
    import numpy as np
    g = np.asarray(grid)
    H, W = g.shape
    
    # Find pattern region: first row/col that is entirely one color (background)
    R = H
    for i in range(H):
        if len(set(g[i].tolist())) == 1:
            R = i
            break
    C = W
    for j in range(W):
        if len(set(g[:, j].tolist())) == 1:
            C = j
            break
    
    pattern = g[:R, :C]
    
    # Find row period
    pr = R
    for p in range(1, R + 1):
        ok = True
        for i in range(R):
            if not np.array_equal(pattern[i], pattern[i % p]):
                ok = False
                break
        if ok:
            pr = p
            break
    
    # Find column period
    pc = C
    for p in range(1, C + 1):
        ok = True
        for j in range(C):
            if not np.array_equal(pattern[:, j], pattern[:, j % p]):
                ok = False
                break
        if ok:
            pc = p
            break
    
    out = np.zeros((H, W), dtype=g.dtype)
    for i in range(H):
        for j in range(W):
            out[i, j] = pattern[i % pr, (j + 1) % pc]
    return out
