def method(grid):
    import numpy as np
    g = grid.copy()
    H, W = g.shape
    
    # Find marker
    mpos = np.argwhere(g == 8)
    mr, mc = mpos[0]
    g[mr, mc] = 0
    
    # Find box (non-zero region)
    nz = np.argwhere(g != 0)
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    
    border = g[r0, c0]
    # Find inner color (any cell strictly inside)
    inner = 0
    for rr in range(r0+1, r1):
        for cc in range(c0+1, c1):
            if g[rr, cc] != 0:
                inner = g[rr, cc]
                break
        if inner != 0:
            break
    
    # Determine direction to marker
    new_r0, new_r1, new_c0, new_c1 = r0, r1, c0, c1
    if mc > c1:
        new_c1 = mc
    elif mc < c0:
        new_c0 = mc
    elif mr > r1:
        new_r1 = mr
    elif mr < r0:
        new_r0 = mr
    
    # Clear old box
    g[r0:r1+1, c0:c1+1] = 0
    
    # Draw new box
    g[new_r0:new_r1+1, new_c0:new_c1+1] = border
    if new_r1 - new_r0 >= 2 and new_c1 - new_c0 >= 2:
        g[new_r0+1:new_r1, new_c0+1:new_c1] = inner
    
    return g
