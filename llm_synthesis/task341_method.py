def method(grid):
    g = grid.copy()
    colors = [c for c in np.unique(g) if c != 0]
    boxes = []
    for c in colors:
        ys, xs = np.where(g == c)
        boxes.append((ys.min(), ys.max(), xs.min(), xs.max()))
    
    (r1a, r1b, c1a, c1b), (r2a, r2b, c2a, c2b) = boxes[0], boxes[1]
    
    col_overlap = not (c1b < c2a or c2b < c1a)
    row_overlap = not (r1b < r2a or r2b < r1a)
    
    if row_overlap and not col_overlap:
        # side-by-side: gap is horizontal (columns between)
        ra = max(r1a, r2a) + 1
        rb = min(r1b, r2b) - 1
        if c1b < c2a:
            ca, cb = c1b + 1, c2a - 1
        else:
            ca, cb = c2b + 1, c1a - 1
        if ra <= rb and ca <= cb:
            g[ra:rb+1, ca:cb+1] = 8
    elif col_overlap and not row_overlap:
        # stacked: gap is vertical (rows between)
        ca = max(c1a, c2a) + 1
        cb = min(c1b, c2b) - 1
        if r1b < r2a:
            ra, rb = r1b + 1, r2a - 1
        else:
            ra, rb = r2b + 1, r1a - 1
        if ra <= rb and ca <= cb:
            g[ra:rb+1, ca:cb+1] = 8
    
    return g
