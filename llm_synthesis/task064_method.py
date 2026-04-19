def method(grid):
    import numpy as np
    g = grid.copy()
    h, w = g.shape
    vals, counts = np.unique(g, return_counts=True)
    bg = vals[np.argmax(counts)]
    
    others = [v for v in vals if v != bg]
    rect_color = None
    marker_color = None
    rr = None
    for v in others:
        cells = np.argwhere(g == v)
        if len(cells) > 1:
            r0, c0 = cells.min(axis=0)
            r1, c1 = cells.max(axis=0)
            if (r1-r0+1)*(c1-c0+1) == len(cells):
                rect_color = v
                rr = (r0, r1, c0, c1)
                continue
        marker_color = v
    
    if rr is None or marker_color is None:
        return g
    
    r0, r1, c0, c1 = rr
    markers = np.argwhere(g == marker_color)
    for mr, mc in markers:
        if c0 <= mc <= c1:
            if mr < r0:
                g[mr:r0, mc] = marker_color
            elif mr > r1:
                g[r1+1:mr+1, mc] = marker_color
        elif r0 <= mr <= r1:
            if mc < c0:
                g[mr, mc:c0] = marker_color
            elif mc > c1:
                g[mr, c1+1:mc+1] = marker_color
    
    return g
