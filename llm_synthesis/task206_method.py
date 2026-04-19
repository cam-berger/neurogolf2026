def method(grid):
    g = grid.copy()
    
    # Find the 5 marker
    fives = np.argwhere(g == 5)
    if len(fives) == 0:
        return g
    fr, fc = fives[0]
    
    # Find bounding box of the shape (non-zero, non-5 cells)
    shape_mask = (g != 0) & (g != 5)
    coords = np.argwhere(shape_mask)
    if len(coords) == 0:
        g[fr, fc] = 0
        return g
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    
    shape = g[r0:r1+1, c0:c1+1].copy()
    h, w = shape.shape
    cr, cc = h // 2, w // 2
    
    # Erase the 5
    g[fr, fc] = 0
    
    # Place the shape so its center aligns with the 5's former position
    for i in range(h):
        for j in range(w):
            v = shape[i, j]
            if v != 0:
                rr = fr - cr + i
                ccol = fc - cc + j
                if 0 <= rr < g.shape[0] and 0 <= ccol < g.shape[1]:
                    g[rr, ccol] = v
    
    return g
