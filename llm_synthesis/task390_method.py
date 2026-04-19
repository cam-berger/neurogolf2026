def method(grid):
    import numpy as np
    g = grid.copy()
    out = g.copy()
    twos = np.argwhere(g == 2)
    rmin, cmin = twos.min(axis=0)
    rmax, cmax = twos.max(axis=0)
    
    # Determine orientation by checking if top boundary row is a full line of 2s
    top_full = all(g[rmin, c] == 2 for c in range(cmin, cmax + 1))
    
    fives = np.argwhere(g == 5)
    # Remove the original 5s
    for r, c in fives:
        out[r, c] = 0
    
    if top_full:
        # Horizontal box: reflect vertically across rmin or rmax
        for r, c in fives:
            if abs(r - rmin) <= abs(r - rmax):
                nr = 2 * rmin - r
            else:
                nr = 2 * rmax - r
            if 0 <= nr < g.shape[0]:
                out[nr, c] = 5
    else:
        # Vertical box: reflect horizontally across cmin or cmax
        for r, c in fives:
            if abs(c - cmin) <= abs(c - cmax):
                nc = 2 * cmin - c
            else:
                nc = 2 * cmax - c
            if 0 <= nc < g.shape[1]:
                out[r, nc] = 5
    
    return out
