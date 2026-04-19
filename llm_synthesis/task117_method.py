def method(grid):
    import numpy as np
    g = grid.copy()
    colors = [c for c in np.unique(g) if c != 0]
    
    pivot_center = None
    other_color = None
    
    for c in colors:
        rs, cs = np.where(g == c)
        r_min, r_max = rs.min(), rs.max()
        c_min, c_max = cs.min(), cs.max()
        cr2 = r_min + r_max
        cc2 = c_min + c_max
        pts = set(zip(rs.tolist(), cs.tolist()))
        # Require full reflection symmetry (both H and V) around center
        h_sym = all((cr2 - r, cc) in pts for r, cc in pts)
        v_sym = all((r, cc2 - cc) in pts for r, cc in pts)
        if h_sym and v_sym:
            pivot_center = (cr2, cc2)
        else:
            other_color = c
    
    if pivot_center is None or other_color is None:
        return g
    
    cr2, cc2 = pivot_center
    rs, cs = np.where(g == other_color)
    H, W = g.shape
    for r, c in zip(rs.tolist(), cs.tolist()):
        for rr, ccc in [(cr2 - r, c), (r, cc2 - c), (cr2 - r, cc2 - c)]:
            if 0 <= rr < H and 0 <= ccc < W:
                g[rr, ccc] = other_color
    
    return g
