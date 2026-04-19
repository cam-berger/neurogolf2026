import numpy as np

def method(grid):
    g = np.array(grid)
    out = np.zeros_like(g)
    
    colors = [c for c in np.unique(g) if c != 0]
    
    shapes = {}
    for c in colors:
        cells = np.argwhere(g == c)
        shapes[c] = cells
    
    def is_rectangle(cells):
        r1, c1 = cells.min(axis=0)
        r2, c2 = cells.max(axis=0)
        return len(cells) == (r2 - r1 + 1) * (c2 - c1 + 1)
    
    target_color = None
    compound_color = None
    for c in colors:
        if is_rectangle(shapes[c]):
            target_color = c
        else:
            compound_color = c
    
    if compound_color is None or target_color is None:
        return g.copy()
    
    cells = shapes[compound_color]
    H, W = g.shape
    mask = np.zeros((H, W), dtype=bool)
    mask[cells[:, 0], cells[:, 1]] = True
    
    r1b, c1b = cells.min(axis=0)
    r2b, c2b = cells.max(axis=0)
    
    best = None
    best_area = 0
    for r1 in range(r1b, r2b + 1):
        for r2 in range(r1, r2b + 1):
            for c1 in range(c1b, c2b + 1):
                for c2 in range(c1, c2b + 1):
                    if mask[r1:r2+1, c1:c2+1].all():
                        area = (r2 - r1 + 1) * (c2 - c1 + 1)
                        if area > best_area:
                            best_area = area
                            best = (r1, c1, r2, c2)
    
    r1, c1, r2, c2 = best
    h = r2 - r1 + 1
    w = c2 - c1 + 1
    
    tcells = shapes[target_color]
    tr1, tc1 = tcells.min(axis=0)
    tr2, tc2 = tcells.max(axis=0)
    
    out[tcells[:, 0], tcells[:, 1]] = target_color
    
    rect_set = set()
    for rr in range(r1, r2 + 1):
        for cc in range(c1, c2 + 1):
            rect_set.add((rr, cc))
    tail = [tuple(x) for x in cells if tuple(x) not in rect_set]
    
    new_r1, new_c1 = r1, c1
    if tail:
        tr = sum(p[0] for p in tail) / len(tail)
        tc = sum(p[1] for p in tail) / len(tail)
        rcr = (r1 + r2) / 2
        rcc = (c1 + c2) / 2
        dr = tr - rcr
        dc = tc - rcc
        if abs(dr) >= abs(dc):
            if dr < 0:
                new_r1 = tr2 + 1
                new_c1 = c1
            else:
                new_r1 = tr1 - h
                new_c1 = c1
        else:
            if dc < 0:
                new_c1 = tc2 + 1
                new_r1 = r1
            else:
                new_c1 = tc1 - w
                new_r1 = r1
    
    out[new_r1:new_r1+h, new_c1:new_c1+w] = compound_color
    
    return out
