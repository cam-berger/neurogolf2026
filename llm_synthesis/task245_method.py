def method(grid):
    import numpy as np
    g = np.array(grid)
    out = g.copy()
    
    # Find 3s - they form a rectangle (4 corners)
    rs, cs = np.where(g == 3)
    r0, r1 = rs.min(), rs.max()
    c0, c1 = cs.min(), cs.max()
    
    # Find 2s bounding box
    rs2, cs2 = np.where(g == 2)
    if len(rs2) == 0:
        return out
    br0, br1 = rs2.min(), rs2.max()
    bc0, bc1 = cs2.min(), cs2.max()
    
    h = br1 - br0 + 1
    w = bc1 - bc0 + 1
    rect_h = r1 - r0 + 1
    rect_w = c1 - c0 + 1
    
    # Target top-left: center shape within rectangle
    tr = r0 + (rect_h - h) // 2
    tc = c0 + (rect_w - w) // 2
    
    # Clear 2s
    out[g == 2] = 0
    
    # Place shape at new location
    for r, c in zip(rs2, cs2):
        nr = tr + (r - br0)
        nc = tc + (c - bc0)
        out[nr, nc] = 2
    
    return out
