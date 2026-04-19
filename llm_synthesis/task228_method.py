def method(grid):
    g = np.array(grid)
    out = g.copy()
    
    # Find bounding box of all non-zero cells (this is the box border)
    nz = np.argwhere(g != 0)
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    
    # Inside corner positions
    tl_in = (r0 + 1, c0 + 1)
    tr_in = (r0 + 1, c1 - 1)
    bl_in = (r1 - 1, c0 + 1)
    br_in = (r1 - 1, c1 - 1)
    
    # Outside corner targets (diagonally opposite)
    tl_out = (r0 - 1, c0 - 1)
    tr_out = (r0 - 1, c1 + 1)
    bl_out = (r1 + 1, c0 - 1)
    br_out = (r1 + 1, c1 + 1)
    
    # Get colors at inside corners
    c_tl = g[tl_in]
    c_tr = g[tr_in]
    c_bl = g[bl_in]
    c_br = g[br_in]
    
    # Clear inside corners
    out[tl_in] = 0
    out[tr_in] = 0
    out[bl_in] = 0
    out[br_in] = 0
    
    # Place at diagonally opposite outside corners
    out[br_out] = c_tl
    out[bl_out] = c_tr
    out[tr_out] = c_bl
    out[tl_out] = c_br
    
    return out
