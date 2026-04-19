def method(grid):
    import numpy as np
    g = grid.copy()
    h, w = g.shape
    colors = {}
    for r in range(h):
        for c in range(w):
            if g[r, c] != 0:
                colors.setdefault(int(g[r, c]), []).append((r, c))
    # arrow color = most cells, marker = fewest (single cell)
    arrow_color = max(colors, key=lambda k: len(colors[k]))
    marker_color = min(colors, key=lambda k: len(colors[k]))
    mr, mc = colors[marker_color][0]
    all_cells = [cell for cs in colors.values() for cell in cs]
    rs = [r for r, c in all_cells]
    cs = [c for r, c in all_cells]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    
    if mr == r0 and mr != r1:
        dr, dc = 1, 0
        start = (r1 + 1, mc)
    elif mr == r1 and mr != r0:
        dr, dc = -1, 0
        start = (r0 - 1, mc)
    elif mc == c0 and mc != c1:
        dr, dc = 0, 1
        start = (mr, c1 + 1)
    elif mc == c1 and mc != c0:
        dr, dc = 0, -1
        start = (mr, c0 - 1)
    else:
        return g
    
    r, c = start
    while 0 <= r < h and 0 <= c < w:
        g[r, c] = marker_color
        r += dr
        c += dc
    return g
